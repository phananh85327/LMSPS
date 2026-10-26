// Amazon-Meta (SNAP) -> filtered_*.csv for dat_generator.py.
//
//   dotnet run -c Release -- <amazon-meta.txt> <raw_out_dir> <filtered_out_dir> [--legacy]
//
// --legacy: no edition merge, placeholder customers kept (the behaviour before the fix).
// hgb/amazon_extract.py is a row-for-row equivalent Python port of this program.
using ImportDBCS;

if (args.Length < 3)
{
    Console.Error.WriteLine("usage: <amazon-meta.txt> <raw_out_dir> <filtered_out_dir> [--legacy]");
    return 1;
}
var legacy = args.Contains("--legacy");
using (var preprocessor = new Preprocessor(args[1])) await preprocessor.ProcessMetaFileAsync(args[0]);
using (var filter = new AmazonGraphFilter(args[2]))
    await filter.ApplyAdvancedFilterAsync(args[1], mergeEditions: !legacy, dropPlaceholders: !legacy);
return 0;
