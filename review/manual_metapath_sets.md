# Manual meta-path sets for the AMAZON user–product graph

Hand-picked meta-path sets to compare against the sets LMSPS finds automatically, for **user →
product link prediction** with HAN (or any meta-path model). Revised 2026-10-26: the first version
assumed P–P / C–C self-loops and unpruned graphs (see section 1). Freeze these sets before you see
any result on the rebuilt data, so the comparison can't be tuned after the fact; suggested
`arch.py` keys are `amazon_lp_m1` … `amazon_lp_m8`.

## 1. The graph and the notation

| Node type | What it is |
|---|---|
| **P** | Product (book, music, DVD, video): the node HAN encodes |
| **U** | User (customer) |
| **C** | Category |

| Relation | Meaning |
|---|---|
| **P–P** | "Customers also bought": Amazon's co-purchase list, built from *all* Amazon customers |
| **U–P** | A user gave the product a positive review (training edges only, in every graph) |
| **P–C** | Product → its leaf category (the rebuilt data links leaves only) |
| **C–C** | Category hierarchy (parent ↔ child) |

A meta-path is written from the target product outwards. `PUP` means product ← user ← product: two
products reviewed by the same user. Every step is a plain relation, with **no self-loops**: `PPUP`
is "reviewed together with an also-bought product" and does not contain `PUP`. That is what a
meta-path means in HAN and metapath2vec, and what LMSPS's search scores since 2026-10-26.

A path and its reverse (`PPUP` / `PUPP`) give the same HAN graph once it is made symmetric; these
sets use the alphabetically smaller key, as the search does.

## 2. How the paths were chosen

1. **One path per kind of evidence.** Products relate in three independent ways:
   - **co-purchase:** Amazon's own behaviour data (`PP`)
   - **co-review:** your users' behaviour (`PUP`)
   - **content:** shared categories (`PCP`)

   Recommender work on this kind of data uses exactly this pattern: one item–user–item path plus one
   item–attribute–item path per attribute.
2. **Short before long.** The HAN paper uses symmetric paths of 1–4 hops. Unpruned, every path with
   two user hops reaches about 60% of all products on the small test graph (table below).
   Your HAN repo's graph builder keeps each product's 20 strongest neighbours after every hop, so a
   long path becomes "the 20 products with the most walks through the strongest intermediate
   nodes". That is a usable graph, but no longer "everything reachable", and it overlaps the
   shorter path it extends (`PUPUP` shares 54% of its edges with `PUP` on the test graph).
3. **No dead paths.** With leaf-only categories, `PCCP` (leaf → parent → products filed directly
   under the parent) leaves 85% of products without a neighbour. It is left out; `PCCCP` (sibling
   leaves through the parent) is the meaningful wider category path.
4. **Sizes match the automatic sets** (4 and 6 paths), so a win can't come from having more paths:
   in the Fleet v2 HAN runs, 6-path sets beat 4-path sets by about 0.012 Hit@10.

### Density of the candidate paths

Small test graph (1,730 products, training edges of split 1, plain relations). Treat it as
illustrative: it keeps few also-bought partners (on the full data 88% of products have at least
one, about 4 on average), so `PP`-paths look emptier here than they are. The "HAN graph" columns
come from your HAN repo's builder (top-20 pruning), which is not reviewed yet.

| Path | Meaning | Unpruned reach: share of pairs (median neighbours) | HAN graph: mean degree | Products isolated |
|---|---|---|---|---|
| `PP` | also bought | 0.01% (0)* | 0.2* | 87%* |
| `PPP` | bought with what's bought with it | 0.01% (0)* | 0.2* | 95%* |
| `PUP` | reviewed by the same user | 3.0% (34) | 26.4 | 0% |
| `PCP` | same leaf category | 3.4% (52) | 23.4 | 2% |
| `PCCP` | products filed under the parent category | 0.08% (0) | 1.4 | 85% |
| `PPUP` | reviewed together with an also-bought product | 1.1% (6)* | 4.2* | 38%* |
| `PCCCP` | sibling categories (via the parent) | 6.2% (92) | 27.1 | 1% |
| `PCPCP` | categories that co-occur with its categories | 19% (329) | 28.4 | 2% |
| `PCPUP` | reviewed together with products of its category | 63% (1,152) | 34.2 | 0% |
| `PUPUP` | two co-review steps | 60% (1,098) | 32.8 | 0% |
| `PUPPUP` | co-review, also-bought, co-review | 18% (262) | 29.4 | 13% |
| `PPUPUP` | also-bought, then two co-review steps | 17% (169)* | 4.6* | 45%* |

\* depends on P–P, which the test graph keeps very little of.

## 3. The sets

| Suggested key | Set | Size | What it tests |
|---|---|---|---|
| `amazon_lp_m1` | `PUP`, `PCP` | 2 | Minimal HAN-paper style: one behaviour path, one attribute path |
| `amazon_lp_m2` | `PP`, `PUP`, `PCP` | 3 | The standard recommender trio |
| `amazon_lp_m3` | `PP`, `PUP`, `PPUP` | 3 | Behaviour only: do categories add anything? |
| `amazon_lp_m4` | `PCP`, `PCCCP`, `PCPCP` | 3 | Content only: how far do categories go alone? |
| `amazon_lp_m5` | `PP`, `PUP`, `PCP`, `PPUP` | 4 | Balanced expert set, same size as the automatic ns 4 sets |
| `amazon_lp_m6` | `PP`, `PPP`, `PUP`, `PCP`, `PPUP`, `PCCCP` | 6 | Balanced expert set, same size as the automatic ns 6 sets |
| `amazon_lp_m7` | `PP`, `PPP`, `PUP`, `PCP` | 4 | No expertise: every path up to 2 hops |
| `amazon_lp_m8` | `PUP`, `PCP`, `PUPUP`, `PCPUP` | 4 | Expert long-range paths |

- **M1** is the floor. If an automatic set can't beat it, the search has not earned its cost.
- **M2 vs M1** shows whether Amazon's co-purchase list adds anything beyond your users' co-reviews.
- **M3** bridges co-purchase and co-review (`PPUP`). M3 ≈ M5 means categories contribute nothing.
- **M4** is a control: content paths are the only ones that work for a product nobody has reviewed
  yet. Expected to be the weakest; its gap to M2 shows how much of the task is behaviour.
- **M5** is the set an expert would hand in: one path per evidence type plus the bridge path.
  **The main comparison for the automatic ns 4 sets.**
- **M6** adds the most meaningful wider versions of the strongest families (`PPP`, `PCCCP`).
  **The main comparison for the automatic ns 6 sets.**
- **M7** takes every path up to 2 hops, with no judgement, like the "maximum hop 2" setting the
  LMSPS paper compares against. If the automatic search can't beat it, the search isn't needed for
  this dataset; if M5/M6 beat it, expertise adds something.
- **M8** gives long paths an expert's fair chance: `PUPUP` (neighbours of co-review neighbours) and
  `PCPUP` (what the audience of this product's category reviews). The automatic sets in Fleet v2
  leaned on two-user-hop paths; M8 shows whether that is the search's insight or just long paths.

## 4. Comparing fairly

- **One graph construction for every set:** same builder, same pruning, `--split-seed 1`.
- **Same split and negatives** as the search. The search chose its paths with the validation
  edges, so compare sets on **test** results; choose among the automatic sets by validation loss.
- **At least 3 run seeds** for every row in the final table; report mean ± std. Differences smaller
  than the seed spread are ties.
- **Popularity row.** With 99 random negatives, ranking products by their number of training
  users already scores well (0.41 Hit@10 on the test graph); every set should be read against it.
- **Random rows:** draw several random sets of the same size from the 36 candidates the search
  chose from, and report the spread over the draws, never a single draw.
- **Compare like with like:** M5 against the ns 4 automatic sets, M6 against ns 6, M7 as the
  reference for both.

## 5. Check the real graphs before training

Your graph builder prints, per path, the mean degree and the number of isolated products. Drop a
path that leaves more than about half the products isolated: it costs a full GAT and contributes
almost nothing.
