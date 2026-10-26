using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace ImportDBCS;

public class Preprocessor : IDisposable
{
    private bool _disposed;

    public const string ProductsFile = "products.csv";
    public const string CategoriesFile = "categories.csv";
    public const string CategoryEdgesFile = "category_edges.csv";
    public const string CategoryPathsFile = "category_paths.csv";
    public const string AlsoBoughtFile = "also_bought.csv";
    public const string ReviewsFile = "reviews.csv";
    public const string UsersFile = "users.csv";

    // categories
    private readonly List<Category> _categories = new List<Category>();
    private readonly List<CategoryEdge> _categoryEdges = new List<CategoryEdge>();
    private readonly HashSet<string> _categorySet = new HashSet<string>(StringComparer.Ordinal);
    private readonly HashSet<string> _categoryEdgeSet = new HashSet<string>(StringComparer.Ordinal);
    private readonly HashSet<string> _users = new HashSet<string>(StringComparer.Ordinal);

    // CSV writers
    private readonly StreamWriter _productsWriter;
    private readonly StreamWriter _prodCatWriter;
    private readonly StreamWriter _prodCatEdgeWriter;
    private readonly StreamWriter _prodCatPathWriter;
    private readonly StreamWriter _alsoBoughtWriter;
    private readonly StreamWriter _reviewsWriter;
    private readonly StreamWriter _usersWriter;

    // Regexes
    private static readonly Regex CategoryInfo = new Regex(@"^\s*(.+?)\s*\[(\d+)\]\s*$", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex ReviewsSummary = new Regex(@"total:\s*(\d+)\s+downloaded:\s*(\d+)\s+avg\s+rating:\s*([0-9.]+)", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex ReviewInfo = new Regex(@"(?<date>\d{4}-\d{1,2}-\d{1,2})\s+(?:cutomer|customer):\s*(?<cust>\S+)\s+rating:\s*(?<rating>[0-9]+(?:\.[0-9]+)?)\s+votes:\s*(?<votes>\d+)\s+helpful:\s*(?<helpful>\d+)", RegexOptions.IgnoreCase | RegexOptions.Compiled);

    public Preprocessor(string outDir)
    {
        this._disposed = false;
        if (Directory.Exists(outDir)) Directory.Delete(outDir, recursive: true);
        Directory.CreateDirectory(outDir);

        // UTF-8 without BOM
        _productsWriter = new StreamWriter(Path.Combine(outDir, ProductsFile), false, new UTF8Encoding(false));
        WriteCsvRow(_productsWriter, ["ID", "ASIN", "Discontinued", "Title", "Group", "SalesRank", "AvgRating"]);

        _prodCatWriter = new StreamWriter(Path.Combine(outDir, CategoriesFile), false, new UTF8Encoding(false));
        WriteCsvRow(_prodCatWriter, ["ID", "Name"]);

        _prodCatEdgeWriter = new StreamWriter(Path.Combine(outDir, CategoryEdgesFile), false, new UTF8Encoding(false));
        WriteCsvRow(_prodCatEdgeWriter, ["ParentID", "ChildID"]);

        _prodCatPathWriter = new StreamWriter(Path.Combine(outDir, CategoryPathsFile), false, new UTF8Encoding(false));
        WriteCsvRow(_prodCatPathWriter, ["ASIN", "No", "Seq", "CategoryID"]);

        _alsoBoughtWriter = new StreamWriter(Path.Combine(outDir, AlsoBoughtFile), false, new UTF8Encoding(false));
        WriteCsvRow(_alsoBoughtWriter, ["ASIN", "AlsoASIN"]);

        _reviewsWriter = new StreamWriter(Path.Combine(outDir, ReviewsFile), false, new UTF8Encoding(false));
        WriteCsvRow(_reviewsWriter, ["ASIN", "Date", "Customer", "Rating", "Votes", "Helpful"]);

        _usersWriter = new StreamWriter(Path.Combine(outDir, UsersFile), false, new UTF8Encoding(false));
        WriteCsvRow(_usersWriter, ["ID", "Name"]);
    }

    public async Task ProcessMetaFileAsync(string path)
    {
        Console.WriteLine($"Processing meta file: {path}");
        using var sr = new StreamReader(path, Encoding.UTF8);
        var count = 0L;

        // accumulate block lines (product entries separated by a blank line)
        var blockLines = new List<string>();
        while (await sr.ReadLineAsync() is { } line)
        {
            // preserve the line text (don't drop category '|' lines)
            if (string.IsNullOrWhiteSpace(line))
            {
                if (blockLines.Count > 0)
                {
                    count++;
                    ProcessBlock(blockLines);
                    blockLines.Clear();
                }
                continue;
            }

            // accumulate
            blockLines.Add(line.Trim());
        }

        // process final block if file doesn't end with blank line
        if (blockLines.Count > 0)
        {
            count++;
            ProcessBlock(blockLines);
            blockLines.Clear();
        }

        foreach (var c in _categories) WriteCsvRow(_prodCatWriter, [c.ID, c.Name]);
        foreach (var ce in _categoryEdges) WriteCsvRow(_prodCatEdgeWriter, [ce.ParentID, ce.ChildID]);

        Console.WriteLine($"Processed {count} product blocks from {path}");
    }

    private void ProcessBlock(List<string> blockLines)
    {
        var id = string.Empty;
        var asin = string.Empty;
        var discontinued = false;
        var title = string.Empty;
        var group = string.Empty;
        var salesRank = 0;

        // Extract basic info
        var line = blockLines.FirstOrDefault(l => l.StartsWith("Id:"));
        if (line != null) id = line.Split("Id:")[1].Trim();
        line = blockLines.FirstOrDefault(l => l.StartsWith("ASIN:"));
        if (line != null) asin = line.Split("ASIN:")[1].Trim();
        line = blockLines.FirstOrDefault(l => l.StartsWith("discontinued product"));
        if (line != null) discontinued = true;

        if (string.IsNullOrWhiteSpace(id) || string.IsNullOrWhiteSpace(asin)) return;
        else if (discontinued)
        {
            WriteCsvRow(_productsWriter, [id, asin, discontinued.ToString(), string.Empty, string.Empty, string.Empty, string.Empty]);
            return;
        }

        // everything after the first "title:" (a title may itself contain "title:")
        line = blockLines.FirstOrDefault(l => l.StartsWith("title:"));
        if (line != null) title = line.Substring("title:".Length).Trim();
        line = blockLines.FirstOrDefault(l => l.StartsWith("group:"));
        if (line != null) group = line.Split("group:")[1].Trim();
        line = blockLines.FirstOrDefault(l => l.StartsWith("salesrank:"));
        if (line != null) int.TryParse(line.Split("salesrank:")[1].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out salesRank);

        // Extract similar prod
        var alsoBought = new List<string>();
        line = blockLines.FirstOrDefault(l => l.StartsWith("similar:"));
        if (line != null)
        {
            var tokens = line.Split(" ", StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length > 2) alsoBought.AddRange(tokens[2..]);
        }

        // Extract categories
        var categoryPaths = new List<List<Category>>();
        line = blockLines.FirstOrDefault(l => l.StartsWith("categories:"));
        if (line != null)
        {
            var index = blockLines.IndexOf(line);
            var linesAfter = blockLines.Skip(index + 1);
            var lines = new List<string>();
            foreach (var l2 in linesAfter)
            {
                if (!l2.StartsWith("|")) break;
                lines.Add(l2);
            }
            foreach (var l in lines)
            {
                var parts = l.Split('|', StringSplitOptions.RemoveEmptyEntries);
                var categoryPath = parts.Select(part =>
                {
                    var category = new Category();
                    if (part.Trim().StartsWith("["))
                    {
                        category.Name = string.Empty;
                        category.ID = part.Trim().Replace("[", string.Empty).Replace("]", string.Empty);
                    }
                    else
                    {
                        var match = CategoryInfo.Match(part.Trim());
                        if (!match.Success || match.Groups.Count != 3) return null;

                        category.Name = match.Groups[1].Value;
                        category.ID = match.Groups[2].Value;
                    }

                    if (_categorySet.Add(category.ID)) _categories.Add(category);
                    return category;
                }).Where(cat => cat != null).Cast<Category>().ToList();

                if (categoryPath.Count > 0) categoryPaths.Add(categoryPath);
                if (categoryPath.Count > 1)
                {
                    for (var i = 1; i < categoryPath.Count; i++)
                    {
                        var categoryEdge = new CategoryEdge
                        {
                            ParentID = categoryPath[i - 1].ID,
                            ChildID = categoryPath[i].ID
                        };
                        var key = $"{categoryEdge.ParentID}|{categoryEdge.ChildID}";
                        if (_categoryEdgeSet.Add(key)) _categoryEdges.Add(categoryEdge);
                    }
                }
            }
        }

        // Extract reviews
        var reviewTotal = 0;
        var reviewDownloaded = 0;
        var reviewAvgRating = 0m;
        var users = new List<User>();
        var reviews = new List<Review>();
        line = blockLines.FirstOrDefault(l => l.StartsWith("reviews:"));
        if (line != null)
        {
            var match = ReviewsSummary.Match(line);
            if (match is { Success: true, Groups.Count: 4 })
            {
                // invariant culture: under e.g. vi-VN "4.5" would otherwise parse as 45
                int.TryParse(match.Groups[1].Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out reviewTotal);
                int.TryParse(match.Groups[2].Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out reviewDownloaded);
                decimal.TryParse(match.Groups[3].Value, NumberStyles.Number, CultureInfo.InvariantCulture, out reviewAvgRating);
            }
            var index = blockLines.IndexOf(line);
            var lines = blockLines.Skip(index + 1);
            foreach (var l in lines)
            {
                match = ReviewInfo.Match(l);
                if (!match.Success) continue;

                var review = new Review
                {
                    ASIN = asin,
                    Date = match.Groups["date"].Value,
                    Customer = match.Groups["cust"].Value,
                    Rating = match.Groups["rating"].Value,
                    Votes = match.Groups["votes"].Value,
                    Helpful = match.Groups["helpful"].Value
                };
                reviews.Add(review);

                // the customer ID is the name: random names made users untraceable
                // in node.dat and changed on every run
                if (_users.Add(review.Customer))
                    users.Add(new User { ID = review.Customer, Name = review.Customer });
            }
        }

        // Write CSVs
        WriteCsvRow(_productsWriter, [id, asin, discontinued.ToString(), title, group, salesRank.ToString(CultureInfo.InvariantCulture), reviewAvgRating.ToString(CultureInfo.InvariantCulture)]);
        foreach (var ab in alsoBought) WriteCsvRow(_alsoBoughtWriter, [asin, ab]);
        for (var a = 0; a < categoryPaths.Count; a++) for (var b = 0; b < categoryPaths[a].Count; b++) WriteCsvRow(_prodCatPathWriter, [asin, a.ToString(), b.ToString(), categoryPaths[a][b].ID]);
        foreach (var rv in reviews) WriteCsvRow(_reviewsWriter, [rv.ASIN, rv.Date, rv.Customer, rv.Rating, rv.Votes, rv.Helpful]);
        foreach (var u in users) WriteCsvRow(_usersWriter, [u.ID, u.Name]);
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