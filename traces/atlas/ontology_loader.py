from __future__ import annotations
from pathlib import Path
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDFS

ATLAS = Namespace("https://w3id.org/atlas/ontology#")


class ATLASGraph:
    def __init__(self, ontology_path: str | Path, vocabularies_path: str | Path):
        self._g = Graph()
        modules_dir = Path(ontology_path).parent / "modules"
        for ttl_file in sorted(modules_dir.glob("*.ttl")):
            self._g.parse(str(ttl_file), format="turtle")
        self.vocab_root = Path(vocabularies_path)

    def resolve_lexicon_files(self, mode_uri: str) -> list[Path]:
        """Return ordered lexicon Paths for a mode (specific → general)."""
        files = []
        current = URIRef(mode_uri)
        visited = set()
        while current and current not in visited:
            visited.add(current)
            lexicon = self._g.value(current, ATLAS.lexiconFile)
            if lexicon:
                files.append(self.vocab_root / str(lexicon))
            parents = [
                p for p in self._g.objects(current, RDFS.subClassOf)
                if isinstance(p, URIRef) and str(p).startswith(str(ATLAS))
            ]
            current = parents[0] if parents else None
        return files

    def label(self, mode_uri: str) -> str:
        labels = list(self._g.objects(URIRef(mode_uri), RDFS.label))
        return str(labels[0]) if labels else mode_uri.split("#")[-1]

    def definition(self, mode_uri: str) -> str | None:
        node = self._g.value(URIRef(mode_uri), ATLAS.definition)
        if node is None:
            return None
        text = str(node).strip()
        return text or None

    def default_severity(self, mode_uri: str) -> float | None:
        node = self._g.value(URIRef(mode_uri), ATLAS.defaultSeverity)
        if node is None:
            return None
        try:
            return float(str(node))
        except (TypeError, ValueError):
            return None

    def curie(self, mode_uri: str) -> str:
        prefix = str(ATLAS)
        if mode_uri.startswith(prefix):
            return f"atlas:{mode_uri.removeprefix(prefix)}"
        return mode_uri

    def is_subclass_of(self, mode_uri: str, ancestor_uri: str) -> bool:
        """Return True iff mode_uri is a (transitive) subclass of ancestor_uri.

        Walks the full rdfs:subClassOf graph (BFS over all parents at each
        level). A class is considered a subclass of itself for this purpose
        — useful when the config maps a folder to a specific leaf mode
        rather than a category.

        BFS rather than single-parent walk so that classes with multiple
        ancestors (`rdfs:subClassOf X, Y`) are correctly classified under
        either branch.
        """
        if mode_uri == ancestor_uri:
            return True
        visited: set[URIRef] = set()
        frontier: list[URIRef] = [URIRef(mode_uri)]
        while frontier:
            node = frontier.pop()
            if node in visited:
                continue
            visited.add(node)
            for parent in self._g.objects(node, RDFS.subClassOf):
                if not isinstance(parent, URIRef) or not str(parent).startswith(str(ATLAS)):
                    continue
                if str(parent) == ancestor_uri:
                    return True
                frontier.append(parent)
        return False