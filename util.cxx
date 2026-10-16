#include "util.hxx"

#include <fstream>

#include <arrow/adapters/orc/adapter.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <TCanvas.h>
#include <TError.h>

void split_path(std::string_view path, std::string *basename,
                std::string *suffix) {
  size_t idx_dot = path.find_last_of(".");
  if (idx_dot == std::string::npos) {
    *basename = path;
    suffix->clear();
  } else {
    *basename = path.substr(0, idx_dot);
    *suffix = path.substr(idx_dot + 1);
  }
}

std::string get_path_suffix(std::string_view path) {
  std::string basename;
  std::string suffix;
  split_path(path, &basename, &suffix);
  return suffix;
}

std::string get_path_basename(std::string_view path) {
  std::string basename;
  std::string suffix;
  split_path(path, &basename, &suffix);
  return basename;
}

FileFormat get_file_format(std::string_view suffix) {
  if (suffix == "root")
    return FileFormat::rntuple;
  else if (suffix == "orc")
    return FileFormat::orc;
  else if (suffix == "parquet")
    return FileFormat::parquet;
  abort();
}

std::vector<std::string> get_column_names(const std::string &basename) {
  std::vector<std::string> colNames;
  std::ifstream fs(basename + "_columns.txt");
  std::string colName;

  while(fs >> colName) {
    colNames.push_back(colName);
  }

  return colNames;
}

std::shared_ptr<arrow::Table> open_arrow(const std::string &input_path, FileFormat fmt) {
  arrow::MemoryPool *pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::io::RandomAccessFile> input =
      arrow::io::ReadableFile::Open(input_path.c_str()).ValueOrDie();

  if (fmt == FileFormat::orc){
    // Open ORC file reader
    auto maybe_reader = arrow::adapters::orc::ORCFileReader::Open(input, pool);
    auto &reader = maybe_reader.ValueOrDie();

    // Read entire file as a single Arrow table
    auto maybe_table = reader->Read();
    std::shared_ptr<arrow::Table> table = maybe_table.ValueOrDie();
    return table;
  } else if (fmt == FileFormat::parquet) {
    // Open Parquet file reader
    auto reader_result = parquet::arrow::OpenFile(input, pool);
    PARQUET_THROW_NOT_OK(reader_result);
    auto reader = std::move(reader_result).ValueOrDie();

    // Read entire file as a single Arrow table
    PARQUET_ASSIGN_OR_THROW(auto table, reader->ReadTable());
    return table;
  }

  return nullptr;
}

void save_histogram(TH1D *hist, const std::string &output_path) {
  gErrorIgnoreLevel = kWarning;
  auto c = TCanvas("c", "", 800, 700);
  hist->GetYaxis()->SetTitle("N_{Events}");
  hist->DrawCopy();
  c.Modified();
  c.Update();
  c.SaveAs(output_path.c_str());
}
