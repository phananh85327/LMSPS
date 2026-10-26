// CSV record types read and written by Preprocessor.cs / AmazonGraphFilter.cs.
// Every field is kept as the exact text of the source file.
namespace ImportDBCS;

public class Product
{
    public string ID { get; set; } = "";
    public string ASIN { get; set; } = "";
    public string Discontinued { get; set; } = "";
    public string Title { get; set; } = "";
    public string Group { get; set; } = "";
    public string SalesRank { get; set; } = "";
    public string AvgRating { get; set; } = "";
}

public class Category
{
    public string ID { get; set; } = "";
    public string Name { get; set; } = "";
}

public class CategoryEdge
{
    public string ParentID { get; set; } = "";
    public string ChildID { get; set; } = "";
}

public class CategoryPath
{
    public string ASIN { get; set; } = "";
    public string No { get; set; } = "";
    public string Seq { get; set; } = "";
    public string CategoryID { get; set; } = "";
}

public class AlsoBought
{
    public string ASIN { get; set; } = "";
    public string AlsoASIN { get; set; } = "";
}

public class Review
{
    public string ASIN { get; set; } = "";
    public string Date { get; set; } = "";
    public string Customer { get; set; } = "";
    public string Rating { get; set; } = "";
    public string Votes { get; set; } = "";
    public string Helpful { get; set; } = "";
}

public class User
{
    public string ID { get; set; } = "";
    public string Name { get; set; } = "";
}
