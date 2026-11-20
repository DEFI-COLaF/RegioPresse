#!/usr/bin/env python3
"""Identification de langue par liste de Swadesh, avec entrée TXT ou XML-TEI.

Pour chaque texte, on regarde quels items de la liste (par langue) apparaissent
dans le texte. Score d'une langue = items trouvés / items valides (en %).
Sont exclus du dénominateur : les cellules NA ; les formes identiques à l'item français
ne sont écartées que sur demande (--drop-identical-fr) et, avec --french-forms, toute forme
qui existe aussi en français (ex. 'grosse'). Par défaut, tous les items sont conservés.
--french-min-freq / --french-freq-col limitent la neutralisation aux formes françaises
assez fréquentes ; --export-neutralised liste les variantes écartées ; --min-tokens écarte
les textes trop courts. --fr-filter {glotlid,zabanshenas} ajoute une étape 1 : les phrases/vers
jugés français sont exclus avant l'identification. Les colonnes listées dans --ignore-cols (défaut : "n°") ne sont jamais évaluées.

Entrée XML-TEI : l'identification est faite à trois niveaux, pour comparer :
  1. div      : tout le texte de la <div> (sous-<div> comprises) ;
  2. bloc     : chaque <p>, <lg>, <q>, <head>, <signed>... (--block-tags) ;
  3. phrase   : chaque <l> (vers) ; pour les blocs en prose, chaque phrase
                (découpage sur . ! ? …).

Usage :
    python swadesh_lid_xml.py lexique.tsv fiute.xml -o resultats.json
        -> resultats.json (hiérarchique) + resultats.tsv (tableau à plat)
    python swadesh_lid_xml.py lexique.tsv texte.txt -o resultats.json
    python swadesh_lid_xml.py lexique.tsv dossier_de_txt/ -o resultats.json
    python swadesh_lid_xml.py lexique.tsv corpus.txt --lines -o resultats.json
    python swadesh_lid_xml.py lexique.tsv fiute.xml --french-forms lexique383.tsv

Comparaison avec Zabanshenas (nécessite le paquet zabanshenas installé) :
    python swadesh_lid_xml.py lexique.tsv fiute.xml --zabanshenas \\
        --zaban-map "occitan=oci,breton=bre"
  Pour savoir quelles langues le modèle connaît : --zaban-list-langs [lexique.tsv].
  Chaque unité (div, bloc, phrase) reçoit aussi l'étiquette et le score de Zabanshenas.
  --zaban-map associe chaque colonne du lexique à un ou plusieurs codes Zabanshenas
  (séparés par « | ») pour calculer la colonne d'accord Swadesh/Zabanshenas.
"""
import argparse
import csv
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

NA_VALUES = {"", "na", "n/a", "nan", "-", "?"}
# Mot = suite de lettres/chiffres, apostrophes internes conservées (ex. c'hwi, l'eau).
# Le trait d'union sépare les mots (peut-être -> peut, être), à l'identique côté
# lexique et côté texte.
TOKEN_RE = re.compile(r"\w+(?:'\w+)*")
SENT_RE = re.compile(r"(?<=[.!?…])\s+(?=\S)")

DEFAULT_BLOCK_TAGS = "p,lg,q,head,signed,opener,salute,note,l,ab,item,label,dateline,fw,byline,trailer"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).casefold()
    return s.replace("’", "'").replace("`", "'").strip()


def norm_col(s: str) -> str:
    """Normalisation des noms de colonnes ('N°', 'nº', ' n° ' -> 'n°')."""
    return norm(s).replace("º", "°")


def tokenize(s: str) -> list:
    return TOKEN_RE.findall(norm(s))


def size_of(texts, count_types=False):
    """Taille d'un texte (ou d'une liste de segments) : nombre de tokens, ou, avec count_types,
    nombre de FORMES DISTINCTES (chaque forme n'est comptée qu'une fois, tous segments confondus)."""
    if isinstance(texts, str):
        texts = [texts]
    if count_types:
        return len({t for x in texts for t in tokenize(x)})
    return sum(len(tokenize(x)) for x in texts)


def parse_cell(cell, split_re):
    """Cellule -> liste de variantes (tuples de tokens). [] si NA."""
    if cell is None or norm(cell) in NA_VALUES:
        return []
    parts = split_re.split(cell) if split_re else [cell]
    out = []
    for p in parts:
        toks = tuple(tokenize(p))
        if toks and toks not in out:
            out.append(toks)
    return out


def load_french_forms(path, freq_col=None, min_freq=0.0):
    """Liste de formes françaises -> dict {tuple de tokens: fréquence cumulée}.

    Un fichier texte (un mot par ligne) ou un TSV (formes en 1re colonne, ex. Lexique383).
    Avec freq_col (nom d'une colonne de l'en-tête, ex. 'freqfilms2') et min_freq > 0,
    seules les formes dont la fréquence cumulée est >= min_freq sont conservées.
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        lines = f.read().splitlines()
    idx = None
    if freq_col or min_freq:
        header = lines[0].split("\t") if lines else []
        if not freq_col:
            sys.exit(f"--french-min-freq demande --french-freq-col. Colonnes de {path} : {header}")
        if freq_col not in header:
            sys.exit(f"Colonne de fréquence {freq_col!r} introuvable dans {path} : {header}")
        idx = header.index(freq_col)
        lines = lines[1:]
    forms = {}
    for line in lines:
        parts = line.split("\t")
        toks = tuple(tokenize(parts[0]))
        if not toks:
            continue
        fr = 0.0
        if idx is not None and idx < len(parts):
            try:
                fr = float(parts[idx].replace(",", "."))
            except ValueError:
                pass
        forms[toks] = forms.get(toks, 0.0) + fr
    if min_freq:
        total = len(forms)
        forms = {k: v for k, v in forms.items() if v >= min_freq}
        print(f"Formes françaises : {len(forms)}/{total} conservées "
              f"({freq_col} >= {min_freq})", file=sys.stderr)
    return forms


def is_french(variant, french_forms, multiword_all):
    if variant in french_forms:
        return True
    # variante de plusieurs mots : option "tous les mots sont français"
    return multiword_all and len(variant) > 1 and all((t,) in french_forms for t in variant)


def load_lexicon(path, french_col, variant_seps, french_forms=None, multiword_all=False,
                 ignore_cols=("n°",), neutral_log=None, drop_identical=False):
    split_re = re.compile("[" + re.escape(variant_seps) + "]") if variant_seps else None
    french_forms = french_forms or set()
    ignore = {norm_col(c) for c in ignore_cols}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        all_cols = reader.fieldnames
        cols = [c for c in all_cols if norm_col(c) not in ignore]
        skipped = [c for c in all_cols if c not in cols]
        if not cols:
            sys.exit("Aucune colonne exploitable après exclusion de " + repr(skipped))
        fr = french_col or cols[0]  # 1re colonne non ignorée
        if fr not in cols:
            sys.exit(f"Colonne française introuvable : {fr!r} (colonnes : {cols})")
        langs = [c for c in cols if c != fr]
        lex = {l: [] for l in langs}  # lang -> liste d'items (liste de variantes)
        stats = {l: {"gardes": 0, "NA": 0, "= item fr": 0, "forme fr": 0} for l in langs}
        for row in reader:
            fr_forms = set(parse_cell(row[fr], split_re))
            for l in langs:
                variants = parse_cell(row.get(l), split_re)
                if not variants:
                    stats[l]["NA"] += 1
                    continue
                kept, removed = [], []
                for v in variants:
                    if drop_identical and v in fr_forms:
                        removed.append((v, "= item fr"))
                    elif is_french(v, french_forms, multiword_all):
                        removed.append((v, "forme fr"))
                    else:
                        kept.append(v)
                if neutral_log is not None:
                    for v, why in removed:
                        freq = french_forms.get(v, "") if isinstance(french_forms, dict) else ""
                        neutral_log.append((l, row[fr], " ".join(v), why,
                                            "non" if kept else "oui", freq))
                if not kept:
                    all_item = all(why == "= item fr" for _, why in removed)
                    stats[l]["= item fr" if all_item else "forme fr"] += 1
                    continue
                lex[l].append(kept)
                stats[l]["gardes"] += 1
    if skipped:
        print(f"Colonnes ignorées : {skipped} ; colonne française : {fr!r}", file=sys.stderr)
    return lex, stats


def build_vocab(lex):
    vocab = {v for items in lex.values() for variants in items for v in variants}
    maxn = max((len(v) for v in vocab), default=1)
    return vocab, maxn


def found_forms(tokens, vocab, maxn):
    """Ensemble des formes du lexique présentes dans la suite de tokens."""
    found = set()
    for n in range(1, maxn + 1):
        for i in range(len(tokens) - n + 1):
            g = tuple(tokens[i:i + n])
            if g in vocab:
                found.add(g)
    # "l'eau" doit aussi permettre de retrouver "eau"
    for t in tokens:
        if "'" in t:
            for p in t.split("'"):
                if p and (p,) in vocab:
                    found.add((p,))
    return found


def score_text(text, lex, vocab, maxn, min_tokens=0, count_types=False):
    """text : une chaîne, ou une liste de segments (vers, phrases...) dont les
    n-grammes ne se chevauchent pas ; les formes trouvées sont réunies."""
    segments = [text] if isinstance(text, str) else text
    found, n_mots, forms = set(), 0, set()
    for seg in segments:
        toks = tokenize(seg)
        n_mots += len(toks)
        forms.update(toks)
        found |= found_forms(toks, vocab, maxn)
    # Le score Swadesh compte déjà chaque item une seule fois (présence/absence). count_types
    # change la TAILLE du texte (seuils, poids) : formes distinctes au lieu de tokens.
    n_tok = len(forms) if count_types else n_mots
    scores, counts = {}, {}
    for lang, items in lex.items():
        valid = len(items)
        hit = sum(1 for variants in items if any(v in found for v in variants))
        counts[lang] = [hit, valid]
        scores[lang] = round(100 * hit / valid, 2) if valid else None
    ranked = {l: s for l, s in scores.items() if s is not None}
    res = {"n_tokens": n_tok, "n_mots": n_mots, "n_formes": len(forms),
           "scores": scores, "counts": counts, "best": None}
    if ranked:
        top = max(ranked.values())
        winners = [l for l, s in ranked.items() if s == top]
        res["best"] = winners[0] if top > 0 else None  # 0 partout = aucune info
        if top > 0 and len(winners) > 1:
            res["ties"] = winners
    if n_tok < min_tokens:  # texte trop court : conservé dans la sortie mais sans verdict
        res["best"] = None
        res.pop("ties", None)
        res["trop_court"] = True
    return res


def winners_of(res):
    """Ensemble des langues gagnantes (ex aequo compris) ; vide si aucune info."""
    if res["best"] is None:
        return set()
    return set(res.get("ties", [res["best"]]))


# ----------------------------------------------------------------------------
# Zabanshenas (optionnel)
# ----------------------------------------------------------------------------
def parse_zaban(r):
    """Normalise la sortie de Zabanshenas.detect -> {"code", "language", "score"}.

    Accepte : liste de listes de dicts, liste de dicts, dict, (étiquette, score), str.
    S'il y a plusieurs candidats, prend celui qui a le meilleur score.
    """
    while isinstance(r, (list, tuple)) and len(r) == 1 and isinstance(r[0], (list, tuple, dict)):
        r = r[0]
    if isinstance(r, (list, tuple)) and r and all(isinstance(x, dict) for x in r):
        r = max(r, key=lambda x: x.get("score", 0) or 0)
    if isinstance(r, dict):
        if {"code", "language", "label"} & set(r):
            code = r.get("code") or r.get("label") or r.get("language")
            return {"code": code, "language": r.get("language"), "score": r.get("score")}
        if r:  # {code: score}
            code, sc = max(r.items(), key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else 0)
            return {"code": code, "language": None, "score": sc if isinstance(sc, (int, float)) else None}
    if isinstance(r, (list, tuple)) and len(r) == 2 and isinstance(r[0], str):
        return {"code": r[0], "language": None, "score": r[1]}
    if isinstance(r, str):
        return {"code": r, "language": None, "score": None}
    return {"code": None, "language": None, "score": None, "brut": repr(r)[:200]}


def zaban_labels(zab):
    """{code: nom} des classes du modèle Zabanshenas chargé, ou None si introuvable."""
    names = getattr(zab, "languages", None)
    names = names if isinstance(names, dict) else {}
    labels = {}
    try:
        id2 = zab.model.config.id2label
        for i, c in sorted(id2.items(), key=lambda kv: int(kv[0])):
            labels[c] = names.get(c)
    except Exception:
        pass
    if not labels:
        for attr in ("id2label", "labels"):
            v = getattr(zab, attr, None)
            if isinstance(v, dict):
                labels = {str(c): names.get(str(c)) for c in v.values()}
            elif isinstance(v, (list, tuple)):
                labels = {str(c): names.get(str(c)) for c in v}
            if labels:
                break
    if not labels and names:
        labels = dict(names)
    return labels or None


def zaban_coverage(labels, zmap, lex_langs):
    """Messages sur la couverture des colonnes du lexique par les classes du modèle."""
    msgs = []
    for l in lex_langs:
        codes = zmap.get(l)
        if not codes:
            msgs.append(f"  {l}: aucune correspondance dans --zaban-map (pas d'accord calculé)")
            continue
        absent = sorted(c for c in codes if c not in labels)
        if absent and len(absent) == len(codes):
            msgs.append(f"  {l}: AUCUN code de {sorted(codes)} n'existe dans le modèle -> "
                        "variété non couverte, accord toujours 0")
        elif absent:
            msgs.append(f"  {l}: codes absents du modèle : {absent}")
    return msgs


def chunked_detector(detect_one, size):
    """Applique detect_one par fenêtres de `size` mots et agrège par vote pondéré (poids = mots).
    Évite la troncature des modèles sur les textes longs. Score = part du texte (pondérée par la
    probabilité du modèle) attribuée à la langue gagnante."""
    def detect(text):
        words = text.split()
        if len(words) <= size:
            return detect_one(text)
        votes, mass, names, total = {}, {}, {}, 0
        for i in range(0, len(words), size):
            piece = words[i:i + size]
            r = detect_one(" ".join(piece))
            total += len(piece)
            c = r.get("code")
            if c:
                votes[c] = votes.get(c, 0) + len(piece)
                mass[c] = mass.get(c, 0.0) + len(piece) * float(r.get("score") or 0.0)
                names[c] = r.get("language")
        if not votes:
            return {"code": None, "language": None, "score": None}
        win = max(votes, key=votes.get)
        return {"code": win, "language": names[win], "score": mass[win] / total}
    return detect


def make_detector(max_length=None, chunk=0):
    """Renvoie une fonction texte -> {"code","language","score"} basée sur Zabanshenas."""
    try:
        from zabanshenas.zabanshenas import Zabanshenas
    except ImportError:
        sys.exit("Le paquet zabanshenas est introuvable : installe-le (ou lance le script "
                 "dans l'environnement où il fonctionne) pour utiliser --zabanshenas.")
    zab = Zabanshenas()
    if hasattr(zab, "load") and getattr(zab, "model", None) is None:
        try:
            zab.load()
        except Exception:
            pass
    extra = {"max_length": max_length} if max_length else {}
    state = {"all_scores_kw": True}

    def detect(text):
        if not tokenize(text):
            return {"code": None, "language": None, "score": None}
        try:
            if state["all_scores_kw"]:
                try:
                    r = zab.detect(text, return_all_scores=False, **extra)
                except TypeError:  # version sans return_all_scores
                    state["all_scores_kw"] = False
                    r = zab.detect(text, **extra)
            else:
                r = zab.detect(text, **extra)
        except Exception as e:  # texte trop court, etc.
            return {"code": None, "language": None, "score": None, "erreur": str(e)[:200]}
        return parse_zaban(r)
    detect.labels = zaban_labels(zab)
    if chunk and chunk > 0:
        wrapped = chunked_detector(detect, chunk)
        wrapped.labels = detect.labels
        return wrapped
    return detect


def parse_zaban_map(spec):
    """'poitevin=pis|roa-tara,saintongeais=pis' -> {'poitevin': {...}, ...}"""
    out = {}
    for part in (spec or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = {c.strip() for c in v.split("|") if c.strip()}
    return out


def agree_zaban(res, zmap):
    """1/0 si le code Zabanshenas est l'un des codes associés aux langues gagnantes Swadesh."""
    z = res.get("zabanshenas", {}).get("code")
    w = winners_of(res)
    if not zmap or not z or not w:
        return ""
    codes = set().union(*(zmap.get(l, set()) for l in w))
    if not codes:
        return ""
    return 1 if z in codes else 0


# ----------------------------------------------------------------------------
# Étape 1 (optionnelle) : repérer le français pour l'exclure
# ----------------------------------------------------------------------------
def resolve_glotlid_path(spec):
    """Chemin local de model.bin. spec peut être : un fichier, un dossier contenant
    model.bin, un identifiant de dépôt Hugging Face (ex. cis-lmu/glotlid), ou vide
    (-> cis-lmu/glotlid)."""
    spec = spec or "cis-lmu/glotlid"
    p = Path(spec).expanduser()
    if p.is_file():
        return str(p)
    if p.is_dir():
        if (p / "model.bin").is_file():
            return str(p / "model.bin")
        sys.exit(f"Pas de model.bin dans le dossier {p}")
    if re.fullmatch(r"[\w.-]+/[\w.-]+", spec) and not spec.startswith((".", "/", "~")):
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(repo_id=spec, filename="model.bin")
        except Exception as e:
            sys.exit(f"Impossible de télécharger {spec}/model.bin ({str(e)[:150]}). Télécharge "
                     "model.bin depuis https://huggingface.co/cis-lmu/glotlid et passe son "
                     "chemin avec --glotlid-model.")
    sys.exit(f"--glotlid-model : fichier ou dossier introuvable : {spec}")


def make_fr_detector(backend, threshold, fr_codes, glotlid_model=None, zaban_detect=None):
    """Renvoie une fonction texte -> {"code", "p_fr", "francais"}.

    backend : "glotlid" (modèle fastText) ou "zabanshenas" (zaban_detect fourni).
    francais = probabilité du français >= threshold.
    """
    codes = {c.strip() for c in fr_codes.split(",") if c.strip()} if fr_codes else None
    if backend == "zabanshenas":
        codes = codes or {"fra", "fr"}

        def detect_zab(text):
            r = zaban_detect(text)
            code, sc = r.get("code"), r.get("score")
            p = (1.0 if sc is None else float(sc)) if code in codes else 0.0
            return {"code": code, "p_fr": p, "francais": code in codes and p >= threshold}
        return detect_zab

    # --- GlotLID (fastText)
    codes = codes or {"fra_Latn"}
    try:
        import fasttext
    except ImportError:
        sys.exit("Le paquet fasttext est introuvable (pip install fasttext-wheel, ou fasttext).")
    path = resolve_glotlid_path(glotlid_model)
    try:  # supprime l'avertissement sans gravité « load_model does not return... »
        fasttext.FastText.eprint = lambda *a, **k: None
    except Exception:
        pass
    model = fasttext.load_model(path)

    def detect_glot(text):
        text = text.replace("\n", " ")
        try:
            labels, probs = model.predict(text, k=10)
            pairs = list(zip(labels, (float(x) for x in probs)))
        except ValueError:  # incompatibilité fasttext / numpy 2
            pairs = [(l, float(p)) for p, l in model.f.predict(text + "\n", 10, 0.0, "strict")]
        pairs = [(l.replace("__label__", ""), p) for l, p in pairs]
        p_fr = sum(p for l, p in pairs if l in codes)
        top = pairs[0][0] if pairs else None
        return {"code": top, "p_fr": p_fr, "francais": p_fr >= threshold}
    return detect_glot


# ----------------------------------------------------------------------------
# Entrée TXT (comportement d'origine)
# ----------------------------------------------------------------------------
def iter_texts(path, lines):
    p = Path(path)
    files = sorted(p.rglob("*.txt")) if p.is_dir() else [p]
    for f in files:
        content = f.read_text(encoding="utf-8", errors="replace")
        if not lines:
            yield f.name, content
            continue
        for n, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            if "\t" in line:  # format "identifiant<TAB>texte"
                tid, text = line.split("\t", 1)
                yield tid, text
            else:
                yield f"{f.name}:{n}", line


# ----------------------------------------------------------------------------
# Entrée XML-TEI
# ----------------------------------------------------------------------------
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def local(tag):
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


BREAK_TAGS = {"lb", "pb", "cb", "milestone"}  # sauts de ligne/colonne : équivalent d'une espace


def get_text_langs(elem, exclude, lang=None):
    """Texte nettoyé d'un élément et, pour chaque caractère, sa langue (xml:lang hérité).

    - les <lb/>, <pb/>, <cb/>, <milestone/> valent une espace (sinon « a<lb/>vostra » donnerait
      « avostra ») ;
    - « ¬ » (césure OCR) et les espaces qui le suivent sont supprimés : « ré¬ serve » -> « réserve » ;
    - les espaces sont normalisées.
    """
    chars = []

    def add(txt, l):
        if txt:
            chars.extend((c, l) for c in txt)

    def rec(e, inherited):
        own = e.get(XML_LANG) or inherited
        add(e.text, own)
        for c in e:
            t = local(c.tag)
            if isinstance(c.tag, str) and t not in exclude:  # (pas les commentaires / PI)
                rec(c, own)
            if t in BREAK_TAGS:
                add(" ", own)
            add(c.tail, own)
    rec(elem, lang)
    out, i, n = [], 0, len(chars)
    while i < n:
        ch, l = chars[i]
        i += 1
        if ch == "¬":
            while i < n and chars[i][0].isspace():
                i += 1
            continue
        out.append((ch, l))
    res, prev_space = [], True
    for ch, l in out:
        if ch.isspace():
            if not prev_space:
                res.append((" ", l))
            prev_space = True
        else:
            res.append((ch, l))
            prev_space = False
    if res and res[-1][0] == " ":
        res.pop()
    return "".join(c for c, _ in res), [l for _, l in res]


def get_text(elem, exclude):
    """Texte d'un élément (descendants compris), sans les éléments exclus."""
    return get_text_langs(elem, exclude)[0]


def counts_from_labeled(text, langs):
    """{langue: nombre de tokens} d'un texte dont chaque caractère porte une langue."""
    c, i, n = {}, 0, len(text)
    while i < n:
        j = i
        while j < n and langs[j] == langs[i]:
            j += 1
        k = len(tokenize(text[i:j]))
        if k:
            c[langs[i]] = c.get(langs[i], 0) + k
        i = j
    return c


def majority(counts):
    """(langue majoritaire, pureté) d'un dict {langue: tokens}."""
    if not counts:
        return None, None
    lang, n = max(counts.items(), key=lambda kv: kv[1])
    return lang, n / sum(counts.values())


def split_units(block, exclude, lang=None):
    """Unités « phrase » d'un bloc : chaque <l> s'il y en a, sinon chaque phrase.
    Renvoie [(texte, élément <l> ou None, (langue, pureté))] ; langue d'après xml:lang."""
    lines = [e for e in block.iter() if local(e.tag) == "l" and e is not block]
    out = []
    if lines:
        inherited = block.get(XML_LANG) or lang
        for l in lines:
            t, lg = get_text_langs(l, exclude, inherited)
            if tokenize(t):
                out.append((t, l, majority(counts_from_labeled(t, lg))))
    else:
        text, langs = get_text_langs(block, exclude, lang)
        spans, start = [], 0
        for m in SENT_RE.finditer(text):
            spans.append((start, m.start()))
            start = m.end()
        spans.append((start, len(text)))
        for a_, b_ in spans:
            t = text[a_:b_]
            if tokenize(t):
                out.append((t, None, majority(counts_from_labeled(t, langs[a_:b_]))))
    return out


def collect(elem, block_tags, lang=None):
    """([(bloc, langue héritée)], [(sous-div, langue héritée)]) de elem, sans descendre dans
    les blocs ni dans les div. La langue est celle des éléments traversés (xml:lang)."""
    blocks, subdivs = [], []

    def rec(e, lg):
        for c in e:
            t = local(c.tag)
            if t == "div":
                subdivs.append((c, lg))
            elif t in block_tags:
                blocks.append((c, lg))
            else:
                rec(c, c.get(XML_LANG) or lg)
    rec(elem, lang)
    return blocks, subdivs


def parse_xml(path, block_tags, exclude, default_lang=None, root=None):
    """Découpe un XML-TEI en div / blocs / phrases. La langue (xml:lang) est héritée de
    l'élément racine, de <text>, de <body>, des <div> et des éléments traversés ; sans
    xml:lang, elle vaut default_lang (ou None)."""
    if root is None:
        root = ET.parse(path).getroot()
    text_el = next((e for e in root.iter() if local(e.tag) == "text"), None)
    body = next((e for e in root.iter() if local(e.tag) == "body"), root)
    base = default_lang
    for e in (root, text_el, body):
        if e is not None and e.get(XML_LANG):
            base = e.get(XML_LANG)
    divs = []
    counter = [0]

    def build_div(elem, parent, fallback_id=None, lang=None):
        counter[0] += 1
        did = elem.get(XML_ID) or fallback_id or f"div{counter[0]:02d}"
        d_lang = elem.get(XML_LANG) or lang
        d = {"id": did, "type": elem.get("type"), "parent": parent, "lang": d_lang,
             "blocks": [], "_sub": [], "_elem": None if fallback_id == "hors-div" else elem}
        divs.append(d)
        blocks, subdivs = collect(elem, block_tags, d_lang)
        for k, (b, bl) in enumerate(blocks, 1):
            text, tl = get_text_langs(b, exclude, bl)
            if not tokenize(text):
                continue
            triples = split_units(b, exclude, bl)
            bid = f"{did}.b{k:02d}"
            has_l = any(local(e.tag) == "l" for e in b.iter() if e is not b)
            segs = [t for t, _, _ in triples] if has_l else [text]
            b_counts = counts_from_labeled(text, tl)
            b_lang, b_pur = majority(b_counts)
            units = [{"id": f"{bid}.s{i}", "text": t, "lang": lp[0], "purity": lp[1], "_elem": _e}
                     for i, (t, _e, lp) in enumerate(triples, 1)]
            d["blocks"].append({
                "id": bid, "tag": local(b.tag), "text": text, "_segments": segs,
                "lang": b_lang, "purity": b_pur, "lang_counts": b_counts, "units": units,
                "_elem": b,
            })
        d["_sub"] = [build_div(sd, did, lang=sl) for sd, sl in subdivs]
        tot = {}
        for part in d["blocks"] + d["_sub"]:
            for l_, n_ in part["lang_counts"].items():
                tot[l_] = tot.get(l_, 0) + n_
        d["lang_counts"] = tot
        d["lang_majoritaire"], d["purity"] = majority(tot)
        return d

    # texte du body hors de toute <div> : pseudo-div
    orphan_blocks, subdivs = collect(body, block_tags, base)
    for sd, sl in subdivs:
        build_div(sd, None, lang=sl)
    if orphan_blocks:
        pseudo = ET.Element("div")
        pseudo.extend([b for b, _ in orphan_blocks])
        build_div(pseudo, None, fallback_id="hors-div", lang=base)
    return divs


def all_units(d):
    us = [u for b in d["blocks"] for u in b["units"]]
    for sub in d["_sub"]:
        us += all_units(sub)
    return us


def run_xml(args, lex, vocab, maxn, detect=None, fr_detect=None, divs=None):
    block_tags = {t.strip() for t in args.block_tags.split(",") if t.strip()}
    exclude = {t.strip() for t in args.exclude_tags.split(",") if t.strip()}
    if divs is None:
        divs = parse_xml(args.texts, block_tags, exclude, getattr(args, "default_lang", None))

    # Étape 1 : filtre « français », au niveau de la phrase/du vers. Les unités jugées
    # françaises sont retirées des textes servant à scorer blocs et div.
    ct = bool(getattr(args, "count_types", False))
    fr_info = {}
    if fr_detect:
        for d in divs:
            for b in d["blocks"]:
                for u in b["units"]:
                    if size_of(u["text"], ct) < args.min_tokens:
                        fr_info[u["id"]] = {"juge": False, "francais": False, "code": None, "p_fr": None}
                    else:
                        r = fr_detect(u["text"])
                        fr_info[u["id"]] = {"juge": True, "francais": r["francais"],
                                            "code": r["code"], "p_fr": r["p_fr"]}

    def kept(b):
        """Segments conservés d'un bloc après le filtre français."""
        if not fr_detect:
            return b["_segments"]
        return [u["text"] for u in b["units"] if not fr_info[u["id"]]["francais"]]

    def div_segs(d):
        segs = [x for b in d["blocks"] for x in kept(b)]
        for sub in d["_sub"]:
            segs += div_segs(sub)
        return segs

    def score(x, text, excluded=False):
        res = score_text(x, lex, vocab, maxn, args.min_tokens, ct)
        if excluded:
            res["best"] = None
            res.pop("ties", None)
            res["exclu_fr"] = True
        if detect:
            res["zabanshenas"] = ({"code": None, "language": None, "score": None, "ecarte": True}
                                  if res.get("trop_court") or res.get("exclu_fr") else detect(text))
        return res

    rows, out_divs = [], []
    for d in divs:
        segs = div_segs(d)
        us = all_units(d)
        n_fr = sum(1 for u in us if fr_info.get(u["id"], {}).get("francais"))
        d_res = score(segs, " ".join(segs), excluded=bool(fr_detect and us and n_fr == len(us)))
        if fr_detect:
            d_res["fr_filtre"] = {"phrases": len(us), "phrases_francaises": n_fr}
        out_d = {"id": d["id"], "type": d["type"], "parent": d["parent"],
                 "niveau": "div", "resultat": d_res, "blocs": []}
        rows.append(("div", d, None, None, d_res, None, None))
        for b in d["blocks"]:
            bsegs = kept(b)
            nb_fr = sum(1 for u in b["units"] if fr_info.get(u["id"], {}).get("francais"))
            b_res = score(bsegs, " ".join(bsegs),
                          excluded=bool(fr_detect and b["units"] and nb_fr == len(b["units"])))
            if fr_detect:
                b_res["fr_filtre"] = {"phrases": len(b["units"]), "phrases_francaises": nb_fr}
            out_b = {"id": b["id"], "tag": b["tag"], "niveau": "bloc",
                     "texte": b["text"], "resultat": b_res, "phrases": []}
            rows.append(("bloc", d, b, None, b_res, d_res, None))
            for u in b["units"]:
                fi = fr_info.get(u["id"])
                u_res = score(u["text"], u["text"], excluded=bool(fi and fi["francais"]))
                if fi:
                    u_res["fr_filtre"] = fi
                out_b["phrases"].append({"id": u["id"], "niveau": "phrase",
                                         "texte": u["text"], "resultat": u_res})
                rows.append(("phrase", d, b, u, u_res, d_res, b_res))
            out_d["blocs"].append(out_b)
        out_divs.append(out_d)
    return out_divs, rows


def agree(res, ref):
    """1/0 si l'unité a une gagnante qui coïncide / non avec celle de ref ; '' si pas d'info."""
    w, r = winners_of(res), winners_of(ref) if ref else set()
    if not w or not r:
        return ""
    return 1 if w & r else 0


def write_tsv(path, rows, langs, zaban=False, zmap=None, fr=False, size_col="n_tokens"):
    head = ["niveau", "div", "div_type", "bloc", "balise", "phrase", size_col, "trop_court",
            "exclu_fr", "best", "ties"] \
        + [f"score_{l}" for l in langs] \
        + (["fr_code", "fr_p", "n_phrases", "n_phrases_fr"] if fr else []) \
        + ["best_div", "best_bloc", "accord_div", "accord_bloc"] \
        + (["zaban_code", "zaban_langue", "zaban_score", "accord_swadesh_zaban"] if zaban else []) \
        + ["texte"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(head)
        for level, d, b, u, res, d_res, b_res in rows:
            text = (u["text"] if u else b["text"] if b else "")
            text = text[:120].replace("\t", " ")
            fi = res.get("fr_filtre", {})
            if fr and level == "phrase":
                fr_cells = [fi.get("code") or "",
                            "" if fi.get("p_fr") is None else round(fi["p_fr"], 4), "", ""]
            elif fr:
                fr_cells = ["", "", fi.get("phrases", ""), fi.get("phrases_francaises", "")]
            else:
                fr_cells = []
            w.writerow([
                level, d["id"], d["type"] or "", b["id"] if b else "", b["tag"] if b else "",
                u["id"] if u else "", res["n_tokens"], 1 if res.get("trop_court") else 0,
                1 if res.get("exclu_fr") else 0, res["best"] or "",
                "|".join(res.get("ties", [])),
                *["" if res["scores"][l] is None else res["scores"][l] for l in langs],
                *fr_cells,
                (d_res["best"] or "") if d_res and level != "div" else "",
                (b_res["best"] or "") if b_res and level == "phrase" else "",
                agree(res, d_res) if level != "div" else "",
                agree(res, b_res) if level == "phrase" else "",
                *([res["zabanshenas"]["code"] or "", res["zabanshenas"]["language"] or "",
                   "" if res["zabanshenas"]["score"] is None else round(res["zabanshenas"]["score"], 4),
                   agree_zaban(res, zmap)] if zaban else []),
                text,
            ])


def print_summary(rows):
    print("\nRésumé par niveau (accord = la langue gagnante coïncide avec celle du niveau supérieur) :",
          file=sys.stderr)
    print(f"{'niveau':8}{'unités':>8}{'avec info':>11}{'écartées':>10}{'accord/div':>16}{'accord/bloc':>16}",
          file=sys.stderr)
    for lvl in ("div", "bloc", "phrase"):
        sel = [r for r in rows if r[0] == lvl]
        info = [r for r in sel if r[4]["best"] is not None]
        cells = []
        for ref_idx in (5, 6):
            if lvl == "div" or (lvl == "bloc" and ref_idx == 6):
                cells.append("-")
                continue
            vals = [agree(r[4], r[ref_idx]) for r in sel]
            vals = [v for v in vals if v != ""]
            cells.append(f"{sum(vals)}/{len(vals)} ({100 * sum(vals) / len(vals):.0f}%)" if vals else "n/a")
        print(f"{lvl:8}{len(sel):>8}{len(info):>11}{sum(1 for r in sel if r[4].get('trop_court') or r[4].get('exclu_fr')):>10}{cells[0]:>16}{cells[1]:>16}", file=sys.stderr)


def print_fr_summary(rows, backend, threshold):
    ph = [r[4] for r in rows if r[0] == "phrase"]
    jug = [r for r in ph if r.get("fr_filtre", {}).get("juge")]
    fr = [r for r in jug if r["fr_filtre"]["francais"]]
    print(f"\nÉtape 1 (filtre français, {backend}, seuil {threshold}) : {len(fr)}/{len(jug)} "
          f"phrases/vers jugés français et exclus ; {len(ph) - len(jug)} non jugés (trop courts). "
          f"Blocs entièrement exclus : {sum(1 for r in rows if r[0] == 'bloc' and r[4].get('exclu_fr'))} ; "
          f"div entièrement exclues : {sum(1 for r in rows if r[0] == 'div' and r[4].get('exclu_fr'))}.",
          file=sys.stderr)


def print_zaban_summary(rows, zmap):
    from collections import Counter
    print("\nZabanshenas : étiquettes les plus fréquentes par niveau :", file=sys.stderr)
    for lvl in ("div", "bloc", "phrase"):
        sel = [r[4] for r in rows if r[0] == lvl]
        cnt = Counter(r["zabanshenas"]["code"] or "(aucune)" for r in sel)
        top = ", ".join(f"{c}:{n}" for c, n in cnt.most_common(6))
        print(f"  {lvl:7} {top}", file=sys.stderr)
    if zmap:
        for lvl in ("div", "bloc", "phrase"):
            vals = [agree_zaban(r[4], zmap) for r in rows if r[0] == lvl]
            vals = [v for v in vals if v != ""]
            if vals:
                print(f"  accord Swadesh/Zabanshenas ({lvl}) : {sum(vals)}/{len(vals)} "
                      f"({100 * sum(vals) / len(vals):.0f}%)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lexicon", nargs="?", help="tableur TSV (1re colonne = français)")
    ap.add_argument("texts", nargs="?", help="fichier .xml (TEI), fichier .txt ou dossier de .txt")
    ap.add_argument("-o", "--output", default="resultats.json")
    ap.add_argument("--tsv", help="tableau à plat (XML) ; défaut : <output>.tsv")
    ap.add_argument("--lines", action="store_true",
                    help="(TXT) une ligne = un texte (format 'id<TAB>texte' accepté)")
    ap.add_argument("--french-col", help="nom de la colonne française (défaut : 1re colonne non ignorée)")
    ap.add_argument("--ignore-cols", default="n°",
                    help="colonnes du lexique à ne pas évaluer, séparées par des virgules (défaut : n°)")
    ap.add_argument("--variant-sep", default="/;,",
                    help="caractères séparant plusieurs variantes dans une cellule ('' pour désactiver)")
    ap.add_argument("--french-forms",
                    help="liste de formes françaises (txt, un mot par ligne, ou TSV avec les "
                         "formes en 1re colonne, ex. Lexique383) : les variantes qui y figurent "
                         "sont neutralisées")
    ap.add_argument("--french-freq-col",
                    help="colonne de fréquence de la liste française (nom dans l'en-tête, "
                         "ex. freqfilms2 ou freqlivres pour Lexique383)")
    ap.add_argument("--french-min-freq", type=float, default=0.0,
                    help="ne neutralise que les formes françaises de fréquence >= ce seuil "
                         "(nécessite --french-freq-col) ; évite que des formes rares du français "
                         "neutralisent des mots de la variété étudiée")
    ap.add_argument("--export-neutralised",
                    help="écrit un TSV des variantes neutralisées (langue, item français, "
                         "variante, motif, item entièrement perdu ?, fréquence française)")
    ap.add_argument("--count-types", action="store_true",
                    help="compter chaque forme une seule fois par texte : les tailles (--min-tokens, "
                         "colonne n_formes) portent sur le nombre de FORMES DISTINCTES au lieu du nombre "
                         "de tokens. Le score Swadesh, lui, compte déjà chaque item une seule fois.")
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="nombre minimal de tokens pour qu'un texte (div, bloc, phrase ou texte "
                         "TXT) soit identifié ; en dessous, aucun verdict (Swadesh ni "
                         "Zabanshenas), l'unité reste dans la sortie avec trop_court=1")
    ap.add_argument("--drop-identical-fr", action="store_true",
                    help="retire du lexique les variantes identiques à l'item français (ex. 'dire' "
                         "pour 'dire'). Par défaut, elles sont CONSERVÉES : le filtre français "
                         "(--fr-filter) écarte en amont les passages en français")
    ap.add_argument("--multiword-all", action="store_true",
                    help="neutralise aussi une variante de plusieurs mots si chacun de ses mots "
                         "est une forme française (défaut : seulement si la séquence exacte "
                         "figure dans la liste)")
    ap.add_argument("--block-tags", default=DEFAULT_BLOCK_TAGS,
                    help=f"(XML) éléments formant le niveau « bloc » (défaut : {DEFAULT_BLOCK_TAGS})")
    ap.add_argument("--exclude-tags", default="",
                    help="(XML) éléments dont le texte est ignoré, ex. 'head,note,signed'")
    ap.add_argument("--zabanshenas", action="store_true",
                    help="ajoute pour chaque unité la prédiction du modèle Zabanshenas")
    ap.add_argument("--zaban-map", default="",
                    help="correspondance colonne du lexique -> code(s) Zabanshenas, ex. "
                         "'poitevin=pis,saintongeais=pis|roa-tara' (pour la colonne d'accord)")
    ap.add_argument("--zaban-list-langs", action="store_true",
                    help="affiche les classes (codes et noms) du modèle Zabanshenas puis s'arrête ; "
                         "si un lexique est donné, indique les colonnes sans classe correspondante "
                         "(utilise --zaban-map)")
    ap.add_argument("--zaban-chunk", type=int, default=0,
                    help="applique Zabanshenas par fenêtres de N mots (vote pondéré) au lieu d'un seul "
                         "appel sur tout le texte ; 0 = désactivé (défaut)")
    ap.add_argument("--zaban-max-length", type=int,
                    help="max_length transmis à Zabanshenas.detect (défaut : celui du modèle)")
    ap.add_argument("--fr-filter", choices=["glotlid", "zabanshenas"],
                    help="étape 1 : détecter le français phrase par phrase (ou vers par vers) et "
                         "l'exclure de l'identification Swadesh (les unités jugées françaises "
                         "sont aussi retirées des textes des blocs et des div)")
    ap.add_argument("--fr-threshold", type=float, default=0.8,
                    help="probabilité minimale du français pour exclure une unité (défaut : 0.8)")
    ap.add_argument("--fr-codes", default="",
                    help="étiquette(s) du français pour le modèle, séparées par des virgules "
                         "(défaut : fra_Latn pour GlotLID ; fra,fr pour Zabanshenas)")
    ap.add_argument("--glotlid-model",
                    help="chemin de model.bin de GlotLID (défaut : téléchargement depuis "
                         "Hugging Face via huggingface_hub)")
    args = ap.parse_args()

    if args.zaban_list_langs:
        det = make_detector(args.zaban_max_length, args.zaban_chunk)
        labels = det.labels
        if not labels:
            sys.exit("Impossible de lire la liste des classes dans cette version de Zabanshenas. "
                     "Inspecte l'objet : print([a for a in dir(Zabanshenas()) if not a.startswith('_')]) "
                     "puis cherche model.config.id2label.")
        print(f"{len(labels)} classes dans le modèle :")
        for code, name in sorted(labels.items()):
            print(f"{code}\t{name or ''}")
        if args.lexicon:
            with open(args.lexicon, encoding="utf-8-sig", newline="") as f:
                cols = next(csv.reader(f, delimiter="\t"))
            ign = {norm_col(c) for c in args.ignore_cols.split(",") if c.strip()}
            cols = [c for c in cols if norm_col(c) not in ign]
            fr_c = args.french_col or cols[0]
            lex_langs = [c for c in cols if c != fr_c]
            print("\nCouverture des colonnes du lexique (d'après --zaban-map) :")
            msgs = zaban_coverage(labels, parse_zaban_map(args.zaban_map), lex_langs)
            print("\n".join(msgs) if msgs else "  toutes les colonnes ont au moins un code présent dans le modèle")
        return
    if not args.lexicon or not args.texts:
        ap.error("lexicon et texts sont requis (sauf avec --zaban-list-langs)")

    french_forms = (load_french_forms(args.french_forms, args.french_freq_col, args.french_min_freq)
                    if args.french_forms else None)
    neutral_log = [] if args.export_neutralised else None
    ignore_cols = [c for c in args.ignore_cols.split(",") if c.strip()]
    lex, stats = load_lexicon(args.lexicon, args.french_col, args.variant_sep,
                              french_forms, args.multiword_all, ignore_cols, neutral_log,
                              args.drop_identical_fr)
    vocab, maxn = build_vocab(lex)
    if neutral_log is not None:
        with open(args.export_neutralised, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["langue", "item_francais", "variante", "motif", "item_perdu", "freq_fr"])
            w.writerows(sorted(neutral_log))
        print(f"{len(neutral_log)} variante(s) neutralisée(s) -> {args.export_neutralised}",
              file=sys.stderr)
    need_zab = args.zabanshenas or args.fr_filter == "zabanshenas"
    zab_detect = make_detector(args.zaban_max_length, args.zaban_chunk) if need_zab else None
    detect = zab_detect if args.zabanshenas else None
    fr_detect = (make_fr_detector(args.fr_filter, args.fr_threshold, args.fr_codes,
                                  args.glotlid_model, zab_detect) if args.fr_filter else None)
    zmap = parse_zaban_map(args.zaban_map)
    if detect and zmap and getattr(detect, "labels", None):
        for m in zaban_coverage(detect.labels, zmap, list(lex)):
            print("Zabanshenas :" + m.strip(), file=sys.stderr)

    if Path(args.texts).suffix.lower() == ".xml":
        out_divs, rows = run_xml(args, lex, vocab, maxn, detect, fr_detect)
        tsv = args.tsv or str(Path(args.output).with_suffix(".tsv"))
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"langues": list(lex), "divs": out_divs}, f, ensure_ascii=False, indent=2)
        write_tsv(tsv, rows, list(lex), bool(detect), zmap, bool(fr_detect),
                  "n_formes" if args.count_types else "n_tokens")
        n = {lvl: sum(1 for r in rows if r[0] == lvl) for lvl in ("div", "bloc", "phrase")}
        print(f"{n['div']} div, {n['bloc']} blocs, {n['phrase']} phrases/vers -> "
              f"{args.output} + {tsv}", file=sys.stderr)
        print_summary(rows)
        if fr_detect:
            print_fr_summary(rows, args.fr_filter, args.fr_threshold)
        if detect:
            print_zaban_summary(rows, zmap)
    else:
        results = {}
        for tid, text in iter_texts(args.texts, args.lines):
            results[tid] = score_text(text, lex, vocab, maxn, args.min_tokens, args.count_types)
            if fr_detect and size_of(text, args.count_types) >= args.min_tokens:
                fi = fr_detect(text)
                results[tid]["fr_filtre"] = fi
                if fi["francais"]:
                    results[tid]["best"] = None
                    results[tid].pop("ties", None)
                    results[tid]["exclu_fr"] = True
            if detect:
                results[tid]["zabanshenas"] = (
                    {"code": None, "language": None, "score": None, "ecarte": True}
                    if results[tid].get("trop_court") or results[tid].get("exclu_fr")
                    else detect(text))
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"{len(results)} texte(s) traité(s) -> {args.output}", file=sys.stderr)

    print("\nItems par langue :", file=sys.stderr)
    for l, st in stats.items():
        print(f"  {l}: {st['gardes']} gardés | neutralisés : {st['= item fr']} (= item fr), "
              f"{st['forme fr']} (forme fr) | {st['NA']} NA", file=sys.stderr)


if __name__ == "__main__":
    main()
