#!/usr/bin/env python3
"""Segment newspaper pages with two Kraken models and inject BnF ALTO TextLines.

Pipeline (per page image found in <issue>/pages/):
  1. Run Kraken baseline segmentation with the LADAS model      -> ALTO A
  2. Run Kraken baseline segmentation with the article model    -> ALTO B
  3. Merge A and B into ONE ALTO. Region labels (<Tags>/<OtherTag>) of both
     models are kept: identical labels are shared, different labels get unique
     IDs, and every TAGREFS in the copied blocks is rewritten accordingly.
  4. Read the TextLines (and children: String, SP, HYP) of the BnF ALTO
     (<issue>/alto/*), rescale their coordinates to the image size, move them
     into the target ALTO namespace and import the styles they reference.
  5. Insert each BnF line in the segmentation TextBlock(s) it falls in.
  6. Validate (unique IDs, resolvable TAGREFS/STYLEREFS, single namespace)
     and write <issue>/output/<image_stem>.xml.

No OCR is performed.
"""

from __future__ import annotations

import argparse
import copy
import re
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path

from lxml import etree
from PIL import Image

try:  # shapely is a Kraken dependency, so it is normally available
    from shapely.geometry import Polygon
    from shapely.geometry import box as shapely_box

    HAVE_SHAPELY = True
except ImportError:  # pragma: no cover
    HAVE_SHAPELY = False


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# --------------------------------------------------------------------------
# Generic XML helpers
# --------------------------------------------------------------------------
def local_name(element):
    return etree.QName(element).localname


def is_element(node):
    return isinstance(node.tag, str)


def find_first(root, name):
    return next(
        (e for e in root.iter() if is_element(e) and local_name(e) == name),
        None,
    )


def find_all(root, name):
    return [e for e in root.iter() if is_element(e) and local_name(e) == name]


def child_named(parent, name):
    return next(
        (c for c in parent if is_element(c) and local_name(c) == name), None
    )


def namespace_of(root):
    return etree.QName(root).namespace


def qname(ns, name):
    return f"{{{ns}}}{name}" if ns else name


def collect_ids(root):
    return {e.get("ID") for e in root.iter() if is_element(e) and e.get("ID")}


def clone_into_namespace(src, ns):
    """Rebuild an element tree inside namespace `ns` with no stale nsmap.

    Needed because BnF uses ALTO v3 and Kraken ALTO v4; a plain deepcopy would
    leave foreign-namespace elements in the output document.
    """
    new = etree.Element(qname(ns, local_name(src)), nsmap={None: ns} if ns else None)
    for key, value in src.attrib.items():
        new.set(key, value)
    for child in src:
        if is_element(child):
            new.append(clone_into_namespace(child, ns))
    return new


def run_command(command):
    print("\n$", " ".join(str(p) for p in command), flush=True)
    result = subprocess.run(
        [str(p) for p in command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): "
            + " ".join(str(p) for p in command)
        )


# --------------------------------------------------------------------------
# File discovery
# --------------------------------------------------------------------------
def find_images(data_dir):
    images = []
    for path in sorted(data_dir.rglob("*")):
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and path.parent.name.lower() == "pages"
        ):
            images.append(path)
    return images


def trailing_number(path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def find_bnf_alto(image_path):
    """pages/0001.jpg -> alto/<anything ending in 1>.xml (0001.xml, page_1.xml)."""
    alto_dir = image_path.parent.parent / "alto"
    if not alto_dir.is_dir():
        raise FileNotFoundError(f"BNF ALTO directory not found: {alto_dir}")

    page_number = trailing_number(image_path)
    if page_number is None:
        raise ValueError(f"Cannot infer page number from {image_path.name}")

    matches = [
        p for p in sorted(alto_dir.glob("*.xml"))
        if trailing_number(p) == page_number
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple BNF ALTO files match page {page_number}: "
            + ", ".join(str(p) for p in matches)
        )
    raise FileNotFoundError(f"No BNF ALTO for page {page_number} in {alto_dir}")


# --------------------------------------------------------------------------
# Kraken
# --------------------------------------------------------------------------
def run_kraken(kraken, image_path, out_xml, model, device=None):
    command = [kraken]
    if device:
        command += ["-d", device]
    command += [
        "-a",  # ALTO output
        "-i", image_path.resolve(), out_xml.resolve(),
        "segment", "-bl",
        "-i", model.resolve(),
    ]
    run_command(command)
    if not out_xml.is_file():
        raise RuntimeError(f"Kraken did not produce {out_xml}")


# --------------------------------------------------------------------------
# ALTO structure helpers (Description, Styles, Tags, Layout order)
# --------------------------------------------------------------------------
def ensure_top_level(root, name, before):
    """Return the top-level <name> element, creating it in schema order."""
    existing = child_named(root, name)
    if existing is not None:
        return existing

    new = etree.Element(qname(namespace_of(root), name), nsmap={None: namespace_of(root)})
    position = None
    for index, child in enumerate(root):
        if is_element(child) and local_name(child) in before:
            position = index
            break
    if position is None:
        root.append(new)
    else:
        root.insert(position, new)
    return new


def ensure_styles(root):
    return ensure_top_level(root, "Styles", before=("Tags", "Layout"))


def ensure_tags(root):
    return ensure_top_level(root, "Tags", before=("Layout",))


# --------------------------------------------------------------------------
# Step 3: merge the two segmentations (keeping labels)
# --------------------------------------------------------------------------
def merge_tags(base_root, other_root, used_ids):
    """Copy <Tags> of `other_root` into `base_root`; return {old_id: new_id}.

    - Same (element, LABEL, DESCRIPTION) already present -> reuse its ID.
    - Otherwise keep the ID if free, else allocate the next free numeric ID
      (e.g. TYPE_4) so that different labels never share an ID.
    """
    mapping = {}
    other_tags = child_named(other_root, "Tags")
    if other_tags is None:
        return mapping

    base_tags = ensure_tags(base_root)
    ns = namespace_of(base_root)

    known = {}
    for tag in base_tags:
        if is_element(tag) and tag.get("ID"):
            known[(local_name(tag), tag.get("LABEL"), tag.get("DESCRIPTION"))] = tag.get("ID")

    for tag in other_tags:
        if not is_element(tag) or not tag.get("ID"):
            continue

        old_id = tag.get("ID")
        key = (local_name(tag), tag.get("LABEL"), tag.get("DESCRIPTION"))

        if key in known:
            mapping[old_id] = known[key]
            continue

        if old_id not in used_ids:
            new_id = old_id
        else:
            match = re.match(r"^(.*?)(\d+)$", old_id)
            prefix = match.group(1) if match else f"{old_id}_"
            numbers = [
                int(m.group(2))
                for i in used_ids
                if (m := re.match(r"^(.*?)(\d+)$", i)) and m.group(1) == prefix
            ]
            counter = max(numbers, default=0) + 1
            new_id = f"{prefix}{counter}"
            while new_id in used_ids:
                counter += 1
                new_id = f"{prefix}{counter}"

        used_ids.add(new_id)
        clone = clone_into_namespace(tag, ns)
        clone.set("ID", new_id)
        base_tags.append(clone)

        known[key] = new_id
        mapping[old_id] = new_id

    return mapping


def remap_tagrefs(element, mapping):
    for node in element.iter():
        if not is_element(node):
            continue
        refs = node.get("TAGREFS")
        if refs is None:
            continue
        mapped = [mapping[r] for r in refs.split() if r in mapping]
        if mapped:
            node.set("TAGREFS", " ".join(mapped))
        else:
            del node.attrib["TAGREFS"]


def dedupe_ids(element, existing_ids, prefix="ART_"):
    """Make IDs of a copied subtree unique in the target document."""
    for node in element.iter():
        if not is_element(node):
            continue
        old = node.get("ID")
        if not old:
            continue
        if old not in existing_ids:
            existing_ids.add(old)
            continue
        new, n = f"{prefix}{old}", 1
        while new in existing_ids:
            new = f"{prefix}{old}_{n}"
            n += 1
        node.set("ID", new)
        existing_ids.add(new)


def merge_segmentations(ladas_xml, article_xml, image_path):
    """Merge article-model regions into the LADAS ALTO.

    Returns (tree, ladas_blocks, article_blocks); the block lists contain the
    TextBlock elements of each model, for per-model line assignment.
    """
    parser = etree.XMLParser(remove_blank_text=True)
    ladas_tree = etree.parse(str(ladas_xml), parser)
    article_root = etree.parse(str(article_xml), parser).getroot()
    ladas_root = ladas_tree.getroot()

    ladas_ps = find_first(ladas_root, "PrintSpace")
    article_ps = find_first(article_root, "PrintSpace")
    if ladas_ps is None or article_ps is None:
        raise ValueError("PrintSpace missing in one of the Kraken ALTO files.")

    ns = namespace_of(ladas_root)
    used_ids = collect_ids(ladas_root)

    ladas_blocks = find_all(ladas_ps, "TextBlock")

    # Labels first, so TAGREFS can be remapped on the copied blocks.
    tag_map = merge_tags(ladas_root, article_root, used_ids)

    article_blocks = []
    for child in list(article_ps):
        if not is_element(child):
            continue
        clone = clone_into_namespace(child, ns)
        remap_tagrefs(clone, tag_map)
        dedupe_ids(clone, used_ids)
        ladas_ps.append(clone)
        if local_name(clone) == "TextBlock":
            article_blocks.append(clone)

    # ALTO has no Page@fileName; the image name lives in sourceImageInformation.
    file_name = find_first(ladas_root, "fileName")
    if file_name is not None:
        file_name.text = image_path.name

    print(f"Tag mapping (article model): {tag_map}")
    print(f"LADAS TextBlocks: {len(ladas_blocks)}, article TextBlocks: {len(article_blocks)}")
    return ladas_tree, ladas_blocks, article_blocks


# --------------------------------------------------------------------------
# Step 2: BnF lines -> rescale
# --------------------------------------------------------------------------
def scale_points(points, sx, sy):
    out = []
    for pair in points.split():
        parts = pair.split(",")
        if len(parts) != 2:
            out.append(pair)
            continue
        try:
            out.append(f"{round(float(parts[0]) * sx)},{round(float(parts[1]) * sy)}")
        except ValueError:
            out.append(pair)
    return " ".join(out)


def rescale_line_geometry(line, sx, sy):
    for node in line.iter():
        if not is_element(node):
            continue
        for attribute, factor in (
            ("HPOS", sx), ("WIDTH", sx), ("VPOS", sy), ("HEIGHT", sy),
        ):
            value = node.get(attribute)
            if value is None:
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            scaled = round(number * factor)
            if attribute in ("WIDTH", "HEIGHT") and number > 0:
                scaled = max(1, scaled)  # never collapse to zero size
            node.set(attribute, str(scaled))

        if node.get("BASELINE"):
            node.set("BASELINE", scale_points(node.get("BASELINE"), sx, sy))
        if local_name(node) == "Baseline" and node.get("POINTS"):
            node.set("POINTS", scale_points(node.get("POINTS"), sx, sy))


def import_styles(bnf_root, lines, target_root):
    """Import BnF TextStyles referenced by the lines; remap STYLEREFS.

    BnF STYLEREFS (TXT_4...) would otherwise dangle in the merged document.
    """
    used = set()
    for line in lines:
        for node in line.iter():
            used.update((node.get("STYLEREFS") or "").split())

    source = {
        s.get("ID"): s for s in find_all(bnf_root, "TextStyle") if s.get("ID")
    }
    existing = collect_ids(target_root)
    ns = namespace_of(target_root)
    mapping, new_styles = {}, []

    for old in sorted(used):
        style = source.get(old)
        if style is None:
            continue
        new, n = f"BNF_{old}", 1
        while new in existing:
            new = f"BNF_{old}_{n}"
            n += 1
        existing.add(new)
        clone = clone_into_namespace(style, ns)
        clone.set("ID", new)
        new_styles.append(clone)
        mapping[old] = new

    if new_styles:
        styles = ensure_styles(target_root)
        position = next(
            (i for i, c in enumerate(styles)
             if is_element(c) and local_name(c) == "ParagraphStyle"),
            len(styles),
        )
        for offset, style in enumerate(new_styles):
            styles.insert(position + offset, style)

    for line in lines:
        for node in line.iter():
            refs = node.get("STYLEREFS")
            if refs is None:
                continue
            mapped = [mapping[r] for r in refs.split() if r in mapping]
            if mapped:
                node.set("STYLEREFS", " ".join(mapped))
            else:
                del node.attrib["STYLEREFS"]


def load_bnf_lines(bnf_xml, image_path, target_ns):
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.parse(str(bnf_xml), parser).getroot()
    page = find_first(root, "Page")
    if page is None:
        raise ValueError(f"No Page element in BNF ALTO: {bnf_xml}")

    lines = [clone_into_namespace(l, target_ns) for l in find_all(root, "TextLine")]
    if not lines:
        raise ValueError(f"No TextLine elements in {bnf_xml}")

    with Image.open(image_path) as image:
        width, height = image.size

    bnf_w = float(page.get("WIDTH", width))
    bnf_h = float(page.get("HEIGHT", height))
    if bnf_w <= 0 or bnf_h <= 0:
        raise ValueError(f"Invalid BNF page size in {bnf_xml}")

    sx, sy = width / bnf_w, height / bnf_h
    for line in lines:
        rescale_line_geometry(line, sx, sy)
    return root, lines, sx, sy


# --------------------------------------------------------------------------
# Step 3: line -> block assignment
# --------------------------------------------------------------------------
def bounding_box(element):
    try:
        x, y = float(element.get("HPOS")), float(element.get("VPOS"))
        w, h = float(element.get("WIDTH")), float(element.get("HEIGHT"))
        return x, y, x + w, y + h
    except (TypeError, ValueError):
        return None


def parse_points(points):
    result = []
    for pair in points.split():
        parts = pair.split(",")
        if len(parts) == 2:
            try:
                result.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return result


class Region:
    """A TextBlock with its polygon (if available) and bbox."""

    def __init__(self, element):
        self.element = element
        self.poly = None

        shape = child_named(element, "Shape")
        polygon = child_named(shape, "Polygon") if shape is not None else None
        if polygon is not None and HAVE_SHAPELY:
            pts = parse_points(polygon.get("POINTS", ""))
            if len(pts) >= 3:
                poly = Polygon(pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty:
                    self.poly = poly

        self.bbox = bounding_box(element)
        if self.bbox is None and self.poly is not None:
            self.bbox = self.poly.bounds
        if self.bbox is None:
            raise ValueError(f"TextBlock {element.get('ID')} has no geometry")
        if self.poly is None and HAVE_SHAPELY:
            self.poly = shapely_box(*self.bbox)

        self.area = (
            self.poly.area if self.poly is not None
            else (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
        )

    def coverage(self, line_box):
        """Fraction of the line's box lying inside this region."""
        line_area = (line_box[2] - line_box[0]) * (line_box[3] - line_box[1])
        if line_area <= 0:
            return 0.0
        left = max(line_box[0], self.bbox[0])
        top = max(line_box[1], self.bbox[1])
        right = min(line_box[2], self.bbox[2])
        bottom = min(line_box[3], self.bbox[3])
        if right <= left or bottom <= top:
            return 0.0
        if self.poly is not None and HAVE_SHAPELY:
            inter = self.poly.intersection(shapely_box(*line_box)).area
        else:
            inter = (right - left) * (bottom - top)
        return inter / line_area

    def center_distance(self, line_box):
        cx, cy = (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2
        lx, ly = (line_box[0] + line_box[2]) / 2, (line_box[1] + line_box[3]) / 2
        return (cx - lx) ** 2 + (cy - ly) ** 2


def best_region(line_box, regions, min_coverage):
    """Highest coverage wins; ties go to the smaller (more specific) region."""
    best, best_key = None, None
    for region in regions:
        cov = region.coverage(line_box)
        if cov <= 0 or cov < min_coverage:
            continue
        key = (round(cov, 3), -region.area)
        if best_key is None or key > best_key:
            best, best_key = region, key
    return best


def uniquify_ids(element, existing_ids, prefix="BNF_"):
    for node in element.iter():
        if not is_element(node):
            continue
        old = node.get("ID")
        if old is None:
            continue
        new, n = f"{prefix}{old}", 1
        while new in existing_ids:
            new = f"{prefix}{old}_{n}"
            n += 1
        existing_ids.add(new)
        node.set("ID", new)


def insert_lines(
    root, lines, ladas_blocks, article_blocks, mode, min_coverage, keep_kraken_lines
):
    all_blocks = ladas_blocks + article_blocks
    if not all_blocks:
        raise ValueError("Segmentation produced no TextBlocks.")

    # Kraken's own (empty) TextLines are replaced by the BnF ones.
    if not keep_kraken_lines:
        for block in all_blocks:
            for child in list(block):
                if is_element(child) and local_name(child) == "TextLine":
                    block.remove(child)

    ladas = [Region(b) for b in ladas_blocks]
    articles = [Region(b) for b in article_blocks]
    everything = ladas + articles
    existing_ids = collect_ids(root)

    inserted = fallbacks = 0
    per_model = {"ladas": 0, "article": 0}

    for index, line in enumerate(lines, start=1):
        line_box = bounding_box(line)
        if line_box is None:
            raise ValueError(f"BNF TextLine #{index} has invalid coordinates")

        targets = []  # (label, Region)
        if mode == "single":
            region = best_region(line_box, everything, 1e-9)
            if region is not None:
                targets.append(("ladas" if region in ladas else "article", region))
        else:
            for label, group in (("ladas", ladas), ("article", articles)):
                region = best_region(line_box, group, min_coverage)
                if region is not None:
                    targets.append((label, region))

        if not targets:
            region = min(everything, key=lambda r: r.center_distance(line_box))
            targets.append(("ladas" if region in ladas else "article", region))
            fallbacks += 1

        for label, region in targets:
            instance = copy.deepcopy(line)
            uniquify_ids(instance, existing_ids)
            region.element.append(instance)
            per_model[label] += 1
            inserted += 1

    print(f"BNF TextLines extracted: {len(lines)}")
    print(f"TextLine insertions: {inserted} (ladas={per_model['ladas']}, article={per_model['article']})")
    print(f"Lines placed by nearest-block fallback: {fallbacks}")
    return inserted


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate(root):
    problems = []

    ids = Counter(e.get("ID") for e in root.iter() if is_element(e) and e.get("ID"))
    duplicates = [i for i, c in ids.items() if c > 1]
    if duplicates:
        problems.append(f"{len(duplicates)} duplicate IDs, e.g. {duplicates[:5]}")

    tag_ids = {t.get("ID") for t in root.iter() if is_element(t) and local_name(t) in ("OtherTag", "LayoutTag", "StructureTag", "RoleTag", "GlossaryRef") and t.get("ID")}
    style_ids = {s.get("ID") for s in find_all(root, "TextStyle") + find_all(root, "ParagraphStyle")}

    missing_tags, missing_styles = set(), set()
    for node in root.iter():
        if not is_element(node):
            continue
        for ref in (node.get("TAGREFS") or "").split():
            if ref not in tag_ids:
                missing_tags.add(ref)
        for ref in (node.get("STYLEREFS") or "").split():
            if ref not in style_ids:
                missing_styles.add(ref)
    if missing_tags:
        problems.append(f"unresolved TAGREFS: {sorted(missing_tags)}")
    if missing_styles:
        problems.append(f"unresolved STYLEREFS: {sorted(missing_styles)}")

    namespaces = {etree.QName(e).namespace for e in root.iter() if is_element(e)}
    if len(namespaces) > 1:
        problems.append(f"mixed namespaces: {namespaces}")

    if problems:
        raise RuntimeError("Validation failed: " + "; ".join(problems))
    print("Validation OK (unique IDs, TAGREFS/STYLEREFS resolved, single namespace).")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def process_image(image_path, args):
    issue_dir = image_path.parent.parent
    output_dir = issue_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    ladas_xml = output_dir / f"{stem}_ladas.xml"
    article_xml = output_dir / f"{stem}_articles.xml"
    final_xml = output_dir / f"{stem}.xml"
    bnf_xml = find_bnf_alto(image_path)

    print("\n" + "=" * 72)
    print(f"Image: {image_path}\nBNF ALTO: {bnf_xml}\nOutput: {final_xml}")
    print("=" * 72)

    # 1 + 2. Kraken segmentation with both models.
    for out_xml, model in ((ladas_xml, args.ladas_model), (article_xml, args.article_model)):
        if args.reuse_segmentation and out_xml.is_file():
            print(f"Reusing {out_xml}")
        else:
            run_kraken(args.kraken, image_path, out_xml, model, args.device)

    # 3. Merge, keeping labels.
    tree, ladas_blocks, article_blocks = merge_segmentations(
        ladas_xml, article_xml, image_path
    )
    root = tree.getroot()

    # 4. BnF TextLines: extract, move to target namespace, rescale, import styles.
    bnf_root, lines, sx, sy = load_bnf_lines(bnf_xml, image_path, namespace_of(root))
    import_styles(bnf_root, lines, root)
    print(f"BNF coordinate scale: x={sx:.6f}, y={sy:.6f}")

    # 5. Insert into blocks.
    insert_lines(
        root, lines, ladas_blocks, article_blocks,
        mode=args.assign,
        min_coverage=args.min_coverage,
        keep_kraken_lines=args.keep_kraken_lines,
    )

    # 6. Validate and write.
    validate(root)
    tree.write(str(final_xml), encoding="UTF-8", xml_declaration=True, pretty_print=True)
    print(f"Final ALTO: {final_xml}")

    if not (args.keep_work or args.reuse_segmentation):
        for path in (ladas_xml, article_xml):
            path.unlink(missing_ok=True)

    return final_xml


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=Path("1_data_exemple"),
                        help="Root directory containing <issue>/pages and <issue>/alto.")
    parser.add_argument("--ladas-model", type=Path, required=True)
    parser.add_argument("--article-model", type=Path, required=True)
    parser.add_argument("--kraken", default="kraken", help="Kraken executable.")
    parser.add_argument("--device", default=None, help="Kraken device, e.g. cpu or cuda:0.")
    parser.add_argument(
        "--assign", choices=("per-model", "single"), default="per-model",
        help="per-model: each line goes into its best LADAS block AND its best "
             "article block (both layers populated, line duplicated). "
             "single: one block per line (most specific wins).",
    )
    parser.add_argument("--min-coverage", type=float, default=0.5,
                        help="Min fraction of a line inside a block (per-model mode).")
    parser.add_argument("--keep-kraken-lines", action="store_true",
                        help="Keep Kraken's own TextLines in the blocks.")
    parser.add_argument("--keep-work", action="store_true",
                        help="Keep raw Kraken ALTO outputs.")
    parser.add_argument("--reuse-segmentation", action="store_true",
                        help="Reuse existing raw Kraken outputs instead of re-running Kraken.")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    data_dir = args.data.resolve()
    args.ladas_model = args.ladas_model.resolve()
    args.article_model = args.article_model.resolve()

    if not data_dir.is_dir():
        parser.error(f"Data directory not found: {data_dir}")
    for model in (args.ladas_model, args.article_model):
        if not model.is_file():
            parser.error(f"Model not found: {model}")
    if not HAVE_SHAPELY:
        print("WARNING: shapely not found; using bounding boxes only.", file=sys.stderr)

    images = find_images(data_dir)
    if not images:
        parser.error(f"No images found under a pages/ directory in {data_dir}")

    print(f"Images found: {len(images)}")
    ok = failed = 0
    for image_path in images:
        try:
            process_image(image_path, args)
            ok += 1
        except Exception as error:
            failed += 1
            print(f"\nERROR processing {image_path}: {error}", file=sys.stderr)
            traceback.print_exc()
            if not args.continue_on_error:
                raise

    print("\n" + "=" * 72)
    print(f"Succeeded: {ok}  Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
