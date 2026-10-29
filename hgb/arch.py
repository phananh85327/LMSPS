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
    # ---- Fleet v2 final schedule (2026-10-22): 30 seeds per ns, split-seed 1,
    # stage 150, 36 reverse-deduplicated candidates, warm-up 15, exhaustive
    # noise-free selection; 1-SE re-selection (delta 0.010, leave-one-out vote;
    # source: logs/Fleet v2/final_ns4|final_ns6/reselect_d010.json "shortlist").
    # "xN" = number of the 30 runs whose 1-SE pick is this set. "_c" = the
    # assembled path-frequency consensus (the ns most frequent paths), which is
    # not necessarily any run's pick. Core relations {PPUPUP, PUPPUP, PUPUP}
    # appear in nearly every pick; the remaining slots are fillers with ~0
    # marginal effect. Downstream MUST use --split-seed 1. Chat log #18, Add. 9-10.
    # SUPERSEDED (2026-10-26): every Fleet v2 set below was searched on the pre-fix AMAZON
    # data (edition duplicates: 73% of val/test edges had a copy in train; placeholder
    # customer ATVPDKIKX0DER on 61% of products). Re-run the search on the rebuilt data
    # before using them -- review/part1_data_pipeline.md.

    # every path of the 13 shortlist sets below, once: building HAN graphs for this key
    # covers all of them (graph files are per path)
    "amazon_lp_unique_strings": [
        ['PCCPUP', 'PCPPUP', 'PCPUP', 'PCPUPP', 'PPCPUP', 'PPPPUP', 'PPPUP', 'PPPUPP', 'PPUP', 'PPUPP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    # ns 4 shortlist (6 sets); _c is also the most frequent pick
    "amazon_lp_f4_s1": [          # x5, consensus
        ['PPPUPP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    "amazon_lp_f4_s2": [         # x4
        ['PPUPP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    "amazon_lp_f4_s3": [         # x4
        ['PCCPUP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    "amazon_lp_f4_s4": [         # x3
        ['PCPUPP', 'PPPUPP', 'PPUPUP', 'PUPPUP'], []],
    "amazon_lp_f4_s5": [         # x3
        ['PCCPUP', 'PPPUPP', 'PPUPUP', 'PUPPUP'], []],
    "amazon_lp_f4_s6": [         # x2
        ['PCPUP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    # ns 6 shortlist (7 sets); here the consensus (_c) was itself picked only x2,
    # so _s1 (x5) is the better single representative
    "amazon_lp_f6_s1": [         # x5, most frequent pick
        ['PCPPUP', 'PPPPUP', 'PPPUPP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    "amazon_lp_f6_s2": [         # x3
        ['PPCPUP', 'PPPPUP', 'PPPUPP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    "amazon_lp_f6_s3": [         # x2
        ['PPPPUP', 'PPPUPP', 'PPUP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    "amazon_lp_f6_s4": [          # x2, consensus
        ['PPPPUP', 'PPPUPP', 'PPUPP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    "amazon_lp_f6_s5": [         # x2
        ['PPCPUP', 'PPPUPP', 'PPUPP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    "amazon_lp_f6_s6": [         # x2
        ['PPPPUP', 'PPUP', 'PPUPP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    "amazon_lp_f6_s7": [         # x2
        ['PPPPUP', 'PPPUP', 'PPPUPP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    # ---- Fleet v3 (2026-10-29): the search rerun on the rebuilt data (review parts 1-9),
    # from logs/Fleet v3/final_ns4|final_ns6/reselect_d010.txt. Per ns: the first two averaged
    # one-SE sets (simplest first) and the vote's consensus _c; ns 6's one-SE window holds 32
    # near-identical sets (review part 4). The category paths in them reach nothing for ~80% of
    # products (books, music) and are kept as selected. Downstream MUST use --split-seed 1 and
    # the data with this fingerprint.
    # ---- split-seed 1, stage 150, ns 4, num_final 10, alpha-edges val, wstep-kfold 4, dropout 0.5, plain relations, data c5f7bccbf61c1b0f; 30 runs in logs\Fleet v3\final_ns4
    # averaged one-SE set, 18 hops, +0.0000 vs the best averaged score, scored in 5/30 runs
    "amazon_lp_v3_f4_a1": [
        ['PCCPCP', 'PPCCPP', 'PPUPP', 'PUPUP'], []],
    # averaged one-SE set, 19 hops, +0.0010 vs the best averaged score, scored in 5/30 runs
    "amazon_lp_v3_f4_a2": [
        ['PCCPPP', 'PPCCPP', 'PPPUPP', 'PUPUP'], []],
    # path-frequency consensus in 5/30 runs (vote, delta 0.010)
    "amazon_lp_v3_f4_c": [
        ['PCCPP', 'PPUPP', 'PUPPUP', 'PUPUP'], []],
    # ---- split-seed 1, stage 150, ns 6, num_final 10, alpha-edges val, wstep-kfold 4, dropout 0.5, plain relations, data c5f7bccbf61c1b0f; 30 runs in logs\Fleet v3\final_ns6
    # averaged one-SE set, 24 hops, +0.0024 vs the best averaged score, scored in 6/30 runs
    "amazon_lp_v3_f6_a1": [
        ['PCCP', 'PCCPP', 'PPCCPP', 'PPUP', 'PUPPUP', 'PUPUP'], []],
    # averaged one-SE set, 25 hops, +0.0018 vs the best averaged score, scored in 6/30 runs
    "amazon_lp_v3_f6_a2": [
        ['PCCP', 'PCCPCP', 'PPCCPP', 'PPUP', 'PUPPUP', 'PUPUP'], []],
    # path-frequency consensus in 5/30 runs (vote, delta 0.010); also one-SE set a9
    "amazon_lp_v3_f6_c": [
        ['PCCP', 'PCCPUP', 'PPCCPP', 'PPUPP', 'PUPPUP', 'PUPUP'], []],
    # Manual meta-path sets
    "amazon_lp_manual_1": [
        ['PUP', 'PCP'], []],
    "amazon_lp_manual_2": [
        ['PP', 'PUP', 'PCP'], []],
    "amazon_lp_manual_3": [
        ['PP', 'PUP', 'PPUP'], []],
    "amazon_lp_manual_4": [
        ['PCP', 'PCCCP', 'PCPCP'], []],
    "amazon_lp_manual_5": [
        ['PP', 'PUP', 'PCP', 'PPUP'], []],
    "amazon_lp_manual_6": [
        ['PP', 'PUP', 'PCP', 'PPUP', 'PPP', 'PCCCP'], []],
    "amazon_lp_manual_7": [
        ['PP', 'PPP', 'PUP', 'PCP'], []],
    "amazon_lp_manual_8": [
        ['PUP', 'PCP', 'PUPUP', 'PCPUP'], []],
}

# reselect_fleet.py names the vote's consensus <prefix>_c, and the README and the HAN repo
# use those names; the Fleet v2 entries above were pasted with the consensus as a
# numbered set. Same list objects, so either key gives the same paths.
archs["amazon_lp_f4_c"] = archs["amazon_lp_f4_s1"]
archs["amazon_lp_f6_c"] = archs["amazon_lp_f6_s4"]
