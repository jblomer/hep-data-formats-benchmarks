import ROOT
import uproot
import awkward as ak
from pyarrow import orc
import pyarrow.parquet as pq
import argparse


def read_as_ak(dataset_name, dataset_path):
    with uproot.open(dataset_path) as file:
        ds = file[dataset_name]
        ak_array = ds.arrays(library="ak")
    return ak_array


def convert2orc(ak_array, output_name, events_per_cluster, uncompressed=False, mirror_rntuple_settings=False):
    compression = "UNCOMPRESSED" if uncompressed else "ZSTD"
    arrow_table = ak.to_arrow_table(ak_array, extensionarray=False)
    orc.write_table(
        arrow_table,
        output_name,
        compression=compression,
        compression_block_size=1024 * 1024 if mirror_rntuple_settings else 64 * 1024,
        stripe_size=events_per_cluster if mirror_rntuple_settings else 64 * 1024 * 1024
    )


def convert2parquet(ak_array, output_name, events_per_cluster, uncompressed=False, mirror_rntuple_settings=False):
    compression = "NONE" if uncompressed else "ZSTD"
    ak.to_parquet(
        ak_array,
        output_name,
        extensionarray=False,
        compression="NONE" if uncompressed else "ZSTD",
        compression_level=None if uncompressed else 3,
        parquet_metadata_statistics=False,
        data_page_size=1024 * 1024 if mirror_rntuple_settings else None,
        row_group_size=events_per_cluster if mirror_rntuple_settings else 64 * 1024 * 1024
    )


def convert2rntuple(dataset_name, input_path, output_path, uncompressed=False):
    rdf = ROOT.RDataFrame(dataset_name, input_path)
    opts = ROOT.RDF.RSnapshotOptions()
    opts.fCompressionAlgorithm = ROOT.RCompressionSetting.EAlgorithm.kUndefined if uncompressed else ROOT.RCompressionSetting.EAlgorithm.kZSTD
    opts.fCompressionLevel = 0 if uncompressed else 1
    opts.fOutputFormat = ROOT.RDF.ESnapshotOutputFormat.kRNTuple
    opts.fOverwriteIfExists = True
    rdf.Snapshot(dataset_name, output_path, "", opts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="convert", description="convert an RNTuple to ORC or Parquet"
    )
    parser.add_argument("dataset_name", help="name of the dataset")
    parser.add_argument("input_path", help="path to the input RNTuple")
    parser.add_argument("output_path", help="path for the output file")
    parser.add_argument(
        "-o", "--output_mode",
        dest="output_mode",
        choices=["orc", "parquet", "rntuple"],
        default="orc",
        help="format to convert to",
    )
    parser.add_argument(
        "-u",
        "--uncompressed",
        dest="uncompressed",
        action="store_true",
        help="don't compress the data",
    )
    parser.add_argument(
        "-m",
        "--mirror_rntuple",
        dest="mirror_rntuple",
        action="store_true",
        help="mirror rntuple's write options",
    )

    args = parser.parse_args()

    print(
        f"converting {args.dataset_name} from {args.input_path} to {args.output_path} (target format: {args.output_mode}, using rntuple write opts: {args.mirror_rntuple}, uncompressed: {args.uncompressed})"
    )

    if args.output_mode == "rntuple":
        convert2rntuple(args.dataset_name, args.input_path, args.output_path, args.uncompressed)
        exit(0)

    ak_array = read_as_ak(args.dataset_name, args.input_path)
    ntuple = ROOT.RNTupleReader.Open(args.dataset_name, args.input_path)
    desc = ntuple.GetDescriptor()
    events_per_cluster = int(desc.GetNEntries() / desc.GetNClusters())
    if args.output_mode == "orc":
        convert2orc(
            ak_array,
            args.output_path,
            events_per_cluster,
            uncompressed=args.uncompressed,
            mirror_rntuple_settings=args.mirror_rntuple,
        )
    if args.output_mode == "parquet":
        convert2parquet(
            ak_array,
            args.output_path,
            events_per_cluster,
            uncompressed=args.uncompressed,
            mirror_rntuple_settings=args.mirror_rntuple,
        )

    print("---> done!")
