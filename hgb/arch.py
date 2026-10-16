archs = {
    "dblp": [
        ['AP', 'APT', 'APVP', 'APAPA', 'APTPA', 'APTPT', 'APTPV', 'APVPA', 'APVPV', 'APAPAP', 'APAPTP', 'APAPVP',
         'APTPTP', 'APTPVP', 'APVPTP', 'APAPAPV', 'APAPTPA', 'APAPTPV', 'APAPVPA', 'APAPVPT', 'APAPVPV', 'APTPAPA',
         'APTPAPT', 'APTPTPT', 'APTPVPV', 'APVPAPT', 'APVPTPA', 'APVPVPA', 'APVPVPT', 'APVPVPV'], []],
    "imdb": [
        ['M', 'MA', 'MK', 'MAM', 'MDM', 'MKM', 'MAMK', 'MDMK', 'MKMD', 'MDMKM', 'MKMAM', 'MKMDM', 'MKMKM', 'MAMAMK',
         'MAMDMA', 'MAMDMK', 'MAMKMD', 'MAMKMK', 'MDMAMA', 'MDMAMD', 'MDMDMA', 'MDMDMK', 'MDMKMA', 'MKMAMA', 'MKMAMD',
         'MKMAMK', 'MKMDMA', 'MKMDMK', 'MKMKMD', 'MKMKMK'], []],
    "acm": [
        ['PPP', 'PAPP', 'PCPA', 'PCPP', 'PPPC', 'PPPP', 'PAPAP', 'PAPPP', 'PCPAP', 'PCPPA', 'PPAPA', 'PPAPC', 'PPAPP',
         'PAPAPA', 'PAPCPA', 'PAPPAP', 'PAPPCP', 'PAPPPP', 'PCPAPP', 'PCPCPP', 'PCPPAP', 'PCPPPP', 'PPAPAP', 'PPAPCP',
         'PPAPPA', 'PPAPPP', 'PPCPAP', 'PPPAPA', 'PPPCPA', 'PPPPPP'], []],
    # Discovered sets. Record provenance so retrain/downstream can use the SAME
    # --split-seed as the search (the final set is chosen by validation BPR on
    # that split's val edges). "?" = not recorded at search time -- fill in.
    # provenance: split-seed ?, stage ?, ns 5, num_final ?, dedupe-reversed ?
    "amazon_lp_han": [
        ['PCPPUP', 'PPPPUP', 'PPUPUP', 'PUPCCP', 'PUPCPP'], []],
    # provenance: split-seed ?, stage ?, ns 5, num_final ?, dedupe-reversed ?
    "amazon_lp_han2": [
        ['PCPPUP', 'PPPPUP', 'PPUPPP', 'PPUPUP', 'PUPPCP'], []],
    # provenance: split-seed ?, stage ?, ns 5, num_final ?, dedupe-reversed ?
    "amazon_lp_han3": [
        ['PUPP', 'PCPCP', 'PPCCCP', 'PPCPUP', 'PPUPUP'], []],
    # hand-engineered baseline (what an expert would define with NO discovery):
    # co-purchase, same-category, co-reviewed, 2-hop co-purchase, sibling-category.
    # 5 paths to match the discovered amazon_lp_han count, so the ablation
    # "does AUTOMATIC path discovery beat manual paths?" compares path QUALITY at
    # an equal path budget -- see random_paths.py for the matching random row.
    "amazon_lp_manual": [
        ['PP', 'PCP', 'PUP', 'PPP', 'PCCP'], []],
}
