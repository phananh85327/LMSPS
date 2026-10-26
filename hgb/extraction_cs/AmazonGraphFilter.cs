using System.Globalization;
using System.Text;
using CsvHelper;
using CsvHelper.Configuration;

namespace ImportDBCS;

public class AmazonGraphFilter : IDisposable
{
    private bool _disposed;

    public const string FilteredProductsFile = "filtered_products.csv";
    public const string FilteredCategoriesFile = "filtered_categories.csv";
    public const string FilteredCategoryEdgesFile = "filtered_category_edges.csv";
    public const string FilteredCategoryPathsFile = "filtered_category_paths.csv";
    public const string FilteredAlsoBoughtFile = "filtered_also_bought.csv";
    public const string FilteredReviewsFile = "filtered_reviews.csv";
    public const string FilteredUsersFile = "filtered_users.csv";
    public const string EditionMapFile = "filtered_edition_map.csv";

    // Customer IDs that are not people: ATVPDKIKX0DER is Amazon's own seller ID and holds the
    // early anonymous reviews (1995-2004, up to 741 reviews on ONE product); A3UN6WX5RRO2AG holds
    // later anonymous ones (2000-2005, up to 579 on one product). Left in, the first one alone
    // reviewed 61% of the filtered products.
    public static readonly string[] KnownPlaceholderCustomers = ["ATVPDKIKX0DER", "A3UN6WX5RRO2AG"];

    private readonly string _outDir;

    // CSV writers
    private readonly StreamWriter _productsWriter;
    private readonly StreamWriter _prodCatWriter;
    private readonly StreamWriter _prodCatEdgeWriter;
    private readonly StreamWriter _prodCatPathWriter;
    private readonly StreamWriter _alsoBoughtWriter;
    private readonly StreamWriter _reviewsWriter;
    private readonly StreamWriter _usersWriter;

    public AmazonGraphFilter(string outDir)
    {
        this._disposed = false;
        this._outDir = outDir;
        if (Directory.Exists(outDir)) Directory.Delete(outDir, recursive: true);
        Directory.CreateDirectory(outDir);

        // UTF-8 without BOM
        _productsWriter = new StreamWriter(Path.Combine(outDir, FilteredProductsFile), false, new UTF8Encoding(false));
        WriteCsvRow(_productsWriter, ["ID", "ASIN", "Discontinued", "Title", "Group", "SalesRank", "AvgRating"]);

        _prodCatWriter = new StreamWriter(Path.Combine(outDir, FilteredCategoriesFile), false, new UTF8Encoding(false));
        WriteCsvRow(_prodCatWriter, ["ID", "Name"]);

        _prodCatEdgeWriter = new StreamWriter(Path.Combine(outDir, FilteredCategoryEdgesFile), false, new UTF8Encoding(false));
        WriteCsvRow(_prodCatEdgeWriter, ["ParentID", "ChildID"]);

        _prodCatPathWriter = new StreamWriter(Path.Combine(outDir, FilteredCategoryPathsFile), false, new UTF8Encoding(false));
        WriteCsvRow(_prodCatPathWriter, ["ASIN", "No", "Seq", "CategoryID"]);

        _alsoBoughtWriter = new StreamWriter(Path.Combine(outDir, FilteredAlsoBoughtFile), false, new UTF8Encoding(false));
        WriteCsvRow(_alsoBoughtWriter, ["ASIN", "AlsoASIN"]);

        _reviewsWriter = new StreamWriter(Path.Combine(outDir, FilteredReviewsFile), false, new UTF8Encoding(false));
        WriteCsvRow(_reviewsWriter, ["ASIN", "Date", "Customer", "Rating", "Votes", "Helpful"]);

        _usersWriter = new StreamWriter(Path.Combine(outDir, FilteredUsersFile), false, new UTF8Encoding(false));
        WriteCsvRow(_usersWriter, ["ID", "Name"]);
    }

    /// <param name="mergeEditions">merge products that carry the same review lines (editions/formats of one work)</param>
    /// <param name="dropPlaceholders">drop placeholder customers (KnownPlaceholderCustomers + anyone with
    /// at least placeholderMinRepeats reviews on a single product)</param>
    /// <param name="editionMinShared">an edition pair shares at least this many identical review lines ...</param>
    /// <param name="editionMinOverlap">... covering at least this share of the smaller product</param>
    /// <param name="editionMaxRun">review lines shared by more products than this are ignored</param>
    public async Task ApplyAdvancedFilterAsync(string inputDir, decimal posThreshold = 4.0m, int kCore = 5,
        bool mergeEditions = true, bool dropPlaceholders = true, int editionMinShared = 3,
        double editionMinOverlap = 0.5, int editionMaxRun = 50, int placeholderMinRepeats = 50)
    {
        var reviewsPath = Path.Combine(inputDir, Preprocessor.ReviewsFile);

        // ==========================================
        // PHASE 1: LỌC DỮ LIỆU THIẾU (Discontinued Products)
        // ==========================================
        var initialProducts = new List<Product>();
        var validAsins = new HashSet<string>(StringComparer.Ordinal);
        var products = ReadCsv<Product>(Path.Combine(inputDir, Preprocessor.ProductsFile));
        await foreach (var product in products)
        {
            if (!string.IsNullOrEmpty(product.ASIN) && product.Discontinued == "False" && validAsins.Add(product.ASIN))
                initialProducts.Add(product);
        }
        Console.WriteLine($"Done phase 1.");

        // ==========================================
        // PHASE 1b: PLACEHOLDER CUSTOMERS
        // ==========================================
        var placeholders = new HashSet<string>(StringComparer.Ordinal);
        if (dropPlaceholders)
        {
            var perPair = new Dictionary<string, int>(StringComparer.Ordinal);
            await foreach (var r in ReadCsv<Review>(reviewsPath))
            {
                if (string.IsNullOrEmpty(r.Customer) || !validAsins.Contains(r.ASIN)) continue;
                var key = r.Customer + "\t" + r.ASIN;
                perPair[key] = perPair.GetValueOrDefault(key, 0) + 1;
            }
            var total = new Dictionary<string, int>(StringComparer.Ordinal);
            var worst = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var (key, k) in perPair)
            {
                var c = key[..key.IndexOf('\t')];
                total[c] = total.GetValueOrDefault(c, 0) + k;
                if (k > worst.GetValueOrDefault(c, 0)) worst[c] = k;
            }
            foreach (var c in KnownPlaceholderCustomers.Where(total.ContainsKey)) placeholders.Add(c);
            if (placeholderMinRepeats > 0)
                foreach (var (c, k) in worst) if (k >= placeholderMinRepeats) placeholders.Add(c);
            foreach (var c in placeholders)
                Console.WriteLine($"   -> Dropped placeholder customer {c}: {total[c]} review lines, up to {worst[c]} on one product.");
        }

        // ==========================================
        // PHASE 1c: EDITION DUPLICATES
        // Amazon shows one review on every edition/format of a work and SNAP crawled each edition
        // page, so the same review line appears under several ASINs (on the full data 55% of the
        // filtered edges were such copies and 73% of val/test edges had a copy in train). Products
        // sharing >= editionMinShared identical lines covering >= editionMinOverlap of the smaller
        // product are merged into the member with the most review lines (ties: the earliest).
        // ==========================================
        var nProd = initialProducts.Count;
        var productIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var p = 0; p < nProd; p++) productIndex[initialProducts[p].ASIN] = p;
        var repOf = Enumerable.Range(0, nProd).ToArray();
        if (mergeEditions)
        {
            var sigId = new Dictionary<string, int>(StringComparer.Ordinal);
            var keys = new List<long>();
            await foreach (var r in ReadCsv<Review>(reviewsPath))
            {
                if (string.IsNullOrEmpty(r.Customer) || placeholders.Contains(r.Customer) ||
                    !productIndex.TryGetValue(r.ASIN, out var p)) continue;
                var sig = $"{r.Customer}\t{r.Date}\t{r.Rating}\t{r.Votes}\t{r.Helpful}";
                if (!sigId.TryGetValue(sig, out var s)) sigId[sig] = s = sigId.Count;
                keys.Add((long)s * nProd + p);
            }
            sigId.Clear();
            keys.Sort();

            // distinct (line, product) pairs, sorted by line then product
            var nSig = new int[nProd];
            var distinct = new List<long>(keys.Count);
            long prev = -1;
            foreach (var k in keys)
            {
                if (k == prev) continue;
                distinct.Add(k);
                nSig[(int)(k % nProd)]++;
                prev = k;
            }
            keys.Clear();

            var shared = new Dictionary<long, int>();
            for (var i0 = 0; i0 < distinct.Count;)
            {
                var line = distinct[i0] / nProd;
                var i1 = i0;
                while (i1 < distinct.Count && distinct[i1] / nProd == line) i1++;
                var run = i1 - i0;
                if (run >= 2 && run <= editionMaxRun)
                    for (var x = i0; x < i1; x++)
                        for (var y = x + 1; y < i1; y++)
                        {
                            var key = (distinct[x] % nProd) * nProd + distinct[y] % nProd;
                            shared[key] = shared.GetValueOrDefault(key, 0) + 1;
                        }
                i0 = i1;
            }

            var parent = Enumerable.Range(0, nProd).ToArray();
            int Find(int x)
            {
                while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
                return x;
            }
            var nPairs = 0;
            foreach (var (key, k) in shared)
            {
                int a = (int)(key / nProd), b = (int)(key % nProd);
                if (k < editionMinShared || k < editionMinOverlap * Math.Min(nSig[a], nSig[b])) continue;
                int ra = Find(a), rb = Find(b);
                if (ra != rb) parent[Math.Max(ra, rb)] = Math.Min(ra, rb);
                nPairs++;
            }
            var best = new Dictionary<int, int>();
            for (var p = 0; p < nProd; p++)
            {
                var root = Find(p);
                if (!best.TryGetValue(root, out var cur) || nSig[p] > nSig[cur]) best[root] = p;
            }
            for (var p = 0; p < nProd; p++) repOf[p] = best[Find(p)];

            var sizes = repOf.GroupBy(r => r).Select(g => g.Count()).Where(c => c >= 2).ToList();
            Console.WriteLine($"   -> Edition duplicates: {nPairs} product pairs -> {sizes.Count} clusters covering {sizes.Sum()} products.");
        }
        var repAsin = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var p = 0; p < nProd; p++) repAsin[initialProducts[p].ASIN] = initialProducts[repOf[p]].ASIN;

        // ==========================================
        // PHASE 2: CASCADING & LỌC NGƯỠNG TƯƠNG TÁC
        // ==========================================
        var initialReviews = new List<Review>();
        var seenUserProductPairs = new HashSet<string>(StringComparer.Ordinal);
        var reviews = ReadCsv<Review>(reviewsPath);
        await foreach (var review in reviews)
        {
            // Reviews of products dropped in phase 1 must be dropped BEFORE k-core,
            // otherwise k-core counts edges that can never reach the final graph.
            if (string.IsNullOrEmpty(review.Customer) || !validAsins.Contains(review.ASIN)) continue;
            if (placeholders.Contains(review.Customer)) continue;

            if (!decimal.TryParse(review.Rating, NumberStyles.Number, CultureInfo.InvariantCulture, out decimal rating) || rating < posThreshold) continue;

            // If a review has at least 5 votes, but less than 20% of people found it helpful, it is spam/noise. Drop it.
            var isUnhelpful = false;
            if (decimal.TryParse(review.Votes, NumberStyles.Number, CultureInfo.InvariantCulture, out decimal votes) &&
                decimal.TryParse(review.Helpful, NumberStyles.Number, CultureInfo.InvariantCulture, out decimal helpful))
            {
                isUnhelpful = (votes >= 5.0m) && ((helpful / votes) < 0.20m);
            }
            if (isUnhelpful) continue;

            // A user reviewing the same product twice is ONE interaction, not two --
            // duplicates inflate k-core degrees and would duplicate U-P edges downstream.
            // Editions of one work count as the same product (their representative).
            var asin = repAsin[review.ASIN];
            if (seenUserProductPairs.Add($"{review.Customer}\t{asin}"))
            {
                review.ASIN = asin;
                initialReviews.Add(review);
            }
        }
        Console.WriteLine($"Done phase 2.");

        // ==========================================
        // PHASE 3: LỌC SỬ DỤNG K-CORE FILTERING
        // ==========================================
        var changed = true;
        var iteration = 1;
        var nextReviews = initialReviews;
        while (changed)
        {
            changed = false;
            var userCounts = new Dictionary<string, int>(StringComparer.Ordinal);
            var prodCounts = new Dictionary<string, int>(StringComparer.Ordinal);

            foreach (var r in nextReviews)
            {
                userCounts[r.Customer] = userCounts.GetValueOrDefault(r.Customer, 0) + 1;
                prodCounts[r.ASIN] = prodCounts.GetValueOrDefault(r.ASIN, 0) + 1;
            }

            var currentReviews = new List<Review>(nextReviews.Count);
            foreach (var r in nextReviews)
            {
                if (userCounts[r.Customer] >= kCore && prodCounts[r.ASIN] >= kCore) currentReviews.Add(r);
            }

            if (currentReviews.Count < nextReviews.Count)
            {
                changed = true;
                nextReviews = currentReviews;
                Console.WriteLine($"   -> Iteration {iteration}: Graph size reduced to {nextReviews.Count} edges.");
                iteration++;
            }
        }

        // Ordered first-appearance list keeps the output files deterministic.
        var survivingUsers = new HashSet<string>(StringComparer.Ordinal);
        var survivingUsersOrdered = new List<string>();
        foreach (var rv in nextReviews)
        {
            if (survivingUsers.Add(rv.Customer)) survivingUsersOrdered.Add(rv.Customer);
        }
        var survivingProds = nextReviews.Select(rv => rv.ASIN).ToHashSet(StringComparer.Ordinal);
        Console.WriteLine($"-> K-Core Converged. Surviving Products: {survivingProds.Count}");
        Console.WriteLine($"-> K-Core Converged. Surviving Reviews: {nextReviews.Count}");
        Console.WriteLine($"-> K-Core Converged. Surviving Users: {survivingUsers.Count}");
        Console.WriteLine($"Done phase 3.");

        // ==========================================
        // PHASE 4: GHI KẾT QUẢ THEO TẬP SURVIVING NODES
        // ==========================================
        var alsoBoughtSeen = new HashSet<string>(StringComparer.Ordinal);
        var alsoBought = ReadCsv<AlsoBought>(Path.Combine(inputDir, Preprocessor.AlsoBoughtFile));
        await foreach (var ab in alsoBought)
        {
            if (ab.ASIN == ab.AlsoASIN) continue; // self-loop
            var from = repAsin.GetValueOrDefault(ab.ASIN, ab.ASIN);
            var to = repAsin.GetValueOrDefault(ab.AlsoASIN, ab.AlsoASIN);
            if (from == to) continue; // a link between two editions of one work
            if (survivingProds.Contains(from) && survivingProds.Contains(to) &&
                alsoBoughtSeen.Add($"{from}\t{to}"))
                WriteCsvRow(_alsoBoughtWriter, [from, to]);
        }

        var validCategoryIDs = new HashSet<string>(StringComparer.Ordinal);
        var categoryPaths = ReadCsv<CategoryPath>(Path.Combine(inputDir, Preprocessor.CategoryPathsFile));
        if (!mergeEditions)
        {
            await foreach (var cp in categoryPaths)
            {
                if (survivingProds.Contains(cp.ASIN))
                {
                    validCategoryIDs.Add(cp.CategoryID);
                    WriteCsvRow(_prodCatPathWriter, [cp.ASIN, cp.No, cp.Seq, cp.CategoryID]);
                }
            }
        }
        else
        {
            // a merged product gets the distinct category paths of all its editions
            var pathsOf = new Dictionary<string, List<List<string>>>(StringComparer.Ordinal);
            string? lastKey = null;
            List<string>? current = null;
            await foreach (var cp in categoryPaths)
            {
                if (!repAsin.TryGetValue(cp.ASIN, out var rep) || !survivingProds.Contains(rep)) { lastKey = null; continue; }
                var key = cp.ASIN + "\t" + cp.No;
                if (key != lastKey)
                {
                    current = new List<string>();
                    if (!pathsOf.TryGetValue(cp.ASIN, out var list)) pathsOf[cp.ASIN] = list = new List<List<string>>();
                    list.Add(current);
                    lastKey = key;
                }
                current!.Add(cp.CategoryID);
            }
            var membersOf = new Dictionary<int, List<int>>();
            for (var p = 0; p < nProd; p++)
            {
                if (!membersOf.TryGetValue(repOf[p], out var m)) membersOf[repOf[p]] = m = new List<int>();
                m.Add(p);
            }
            for (var p = 0; p < nProd; p++)
            {
                var asin = initialProducts[p].ASIN;
                if (repOf[p] != p || !survivingProds.Contains(asin)) continue;
                var no = 0;
                var seenPaths = new HashSet<string>(StringComparer.Ordinal);
                foreach (var member in membersOf[p])
                {
                    if (!pathsOf.TryGetValue(initialProducts[member].ASIN, out var paths)) continue;
                    foreach (var path in paths)
                    {
                        if (!seenPaths.Add(string.Join("|", path))) continue;
                        for (var seq = 0; seq < path.Count; seq++)
                        {
                            validCategoryIDs.Add(path[seq]);
                            WriteCsvRow(_prodCatPathWriter, [asin, no.ToString(CultureInfo.InvariantCulture), seq.ToString(CultureInfo.InvariantCulture), path[seq]]);
                        }
                        no++;
                    }
                }
            }
        }

        var categoryEdges = ReadCsv<CategoryEdge>(Path.Combine(inputDir, Preprocessor.CategoryEdgesFile));
        await foreach (var ce in categoryEdges)
        {
            if (validCategoryIDs.Contains(ce.ParentID) && validCategoryIDs.Contains(ce.ChildID))
                WriteCsvRow(_prodCatEdgeWriter, [ce.ParentID, ce.ChildID]);
        }

        var categories = ReadCsv<Category>(Path.Combine(inputDir, Preprocessor.CategoriesFile));
        await foreach (var c in categories)
        {
            if (validCategoryIDs.Contains(c.ID)) WriteCsvRow(_prodCatWriter, [c.ID, c.Name]);
        }

        // AvgRating is passed through UNCHANGED from the raw metadata. Recomputing it from
        // the filtered reviews would (a) bias it upward, since everything below posThreshold
        // was already discarded, and (b) derive a product feature from the exact interaction
        // edges the recommender is later asked to predict.
        var nextProducts = initialProducts.Where(p => survivingProds.Contains(p.ASIN)).ToList();
        foreach (var pd in nextProducts)
        {
            WriteCsvRow(_productsWriter, [pd.ID, pd.ASIN, pd.Discontinued, pd.Title, pd.Group, pd.SalesRank, pd.AvgRating]);
        }

        foreach (var rv in nextReviews)
        {
            WriteCsvRow(_reviewsWriter, [rv.ASIN, rv.Date, rv.Customer, rv.Rating, rv.Votes, rv.Helpful]);
        }

        // Users are emitted from the surviving review set itself (names joined from users.csv
        // when available) so that every review edge is guaranteed a user node, even if
        // users.csv is incomplete. Depending on users.csv here previously meant a missing user
        // silently dropped all their edges at .dat generation and broke the k-core guarantee.
        var userNames = new Dictionary<string, string>(StringComparer.Ordinal);
        var users = ReadCsv<User>(Path.Combine(inputDir, Preprocessor.UsersFile));
        await foreach (var u in users)
        {
            if (survivingUsers.Contains(u.ID) && !string.IsNullOrEmpty(u.Name)) userNames[u.ID] = u.Name;
        }
        foreach (var uid in survivingUsersOrdered)
        {
            WriteCsvRow(_usersWriter, [uid, userNames.GetValueOrDefault(uid, uid)]);
        }

        if (mergeEditions)
        {
            // every member of every merged work (representative included), for traceability
            var size = repOf.GroupBy(r => r).ToDictionary(g => g.Key, g => g.Count());
            using var mapWriter = new StreamWriter(Path.Combine(_outDir, EditionMapFile), false, new UTF8Encoding(false));
            WriteCsvRow(mapWriter, ["ASIN", "RepresentativeASIN", "Title"]);
            for (var p = 0; p < nProd; p++)
                if (size[repOf[p]] >= 2)
                    WriteCsvRow(mapWriter, [initialProducts[p].ASIN, initialProducts[repOf[p]].ASIN, initialProducts[p].Title]);
        }

        Console.WriteLine($"Done phase 4.");
    }

    private async IAsyncEnumerable<T> ReadCsv<T>(string path) where T : class
    {
        var cfg = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            HasHeaderRecord = true,
            IgnoreBlankLines = true,
            BadDataFound = null,
            MissingFieldFound = null,
            HeaderValidated = null,
            TrimOptions = TrimOptions.Trim
        };

        using var sr = new StreamReader(path);
        using var csv = new CsvReader(sr, cfg);

        await csv.ReadAsync();
        csv.ReadHeader();
        while (await csv.ReadAsync())
        {
            T record;
            try
            {
                record = csv.GetRecord<T>();
            }
            catch (Exception ex)
            {
                await Console.Error.WriteLineAsync($"CSV parse error in {path}: {ex.Message}");
                continue;
            }
            yield return record;
        }
    }

    private static void WriteCsvRow(StreamWriter w, string[] values)
    {
        var sb = new StringBuilder();
        for (var i = 0; i < values.Length; i++)
        {
            var v = values[i];
            var needQuotes = v.Contains(',') || v.Contains('"') || v.Contains('\n') || v.Contains('\r');
            if (needQuotes)
            {
                v = v.Replace("\"", "\"\"");
                v = '\"' + v + '\"';
            }
            sb.Append(v);
            if (i < values.Length - 1) sb.Append(',');
        }
        w.WriteLine(sb.ToString());
    }

    public void Dispose()
    {
        if (_disposed) return;

        _productsWriter.Dispose();
        _prodCatWriter.Dispose();
        _prodCatEdgeWriter.Dispose();
        _prodCatPathWriter.Dispose();
        _alsoBoughtWriter.Dispose();
        _reviewsWriter.Dispose();
        _usersWriter.Dispose();

        _disposed = true;
    }
}
