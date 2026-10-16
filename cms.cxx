#include <ROOT/RArrowDS.hxx>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RNTupleReader.hxx>

#include <TCanvas.h>
#include <TH1D.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TSystem.h>

#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <arrow/adapters/orc/adapter.h>

#include <chrono>
#include <iostream>
#include <string>

#include "util.hxx"

const std::vector<std::string> columnNames = {
  "nMuon",
  "Muon_charge",
  "Muon_pt",
  "Muon_eta",
  "Muon_phi",
  "Muon_mass"
};

static AnalysisTime_t analysis_orc(const std::string &path,
                                   const std::string &histo_path) {
  auto ts_init = std::chrono::steady_clock::now();

  auto hMass =
      std::make_unique<TH1D>("Dimuon_mass", "Dimuon_mass", 2000, 0.25, 300);

  arrow::MemoryPool *pool = arrow::default_memory_pool();
  auto localFile = arrow::io::ReadableFile::Open(path, pool).ValueOrDie();
  auto reader = arrow::adapters::orc::ORCFileReader::Open(localFile, pool).ValueOrDie();

  auto schema = reader->ReadSchema().ValueOrDie();
  auto nStripes = reader->NumberOfStripes();
  std::shared_ptr<arrow::RecordBatch> recordBatch;

  std::chrono::steady_clock::time_point ts_first =
      std::chrono::steady_clock::now();

  for (std::int64_t stripe = 0; stripe < nStripes; ++stripe) {
    recordBatch = reader->ReadStripe(stripe, columnNames).ValueOrDie();

    auto nMuonArr = std::static_pointer_cast<arrow::Int32Array>(
        recordBatch->GetColumnByName("nMuon"));
    auto muonChargeArr = std::static_pointer_cast<arrow::ListArray>(
        recordBatch->GetColumnByName("Muon_charge"));
    auto muonPtArr = std::static_pointer_cast<arrow::ListArray>(
        recordBatch->GetColumnByName("Muon_pt"));
    auto muonEtaArr = std::static_pointer_cast<arrow::ListArray>(
        recordBatch->GetColumnByName("Muon_eta"));
    auto muonPhiArr = std::static_pointer_cast<arrow::ListArray>(
        recordBatch->GetColumnByName("Muon_phi"));
    auto muonMassArr = std::static_pointer_cast<arrow::ListArray>(
        recordBatch->GetColumnByName("Muon_mass"));

      std::shared_ptr<arrow::Int32Array> nMuonsArray;
      auto rawNMuonVals = nMuonArr->raw_values();
      ROOT::RVec<std::int32_t> nMuons(rawNMuonVals,
                                      rawNMuonVals + nMuonArr->length());

      ROOT::RVec<std::int32_t> muonCharge;
      ROOT::RVec<float> muonPt, muonEta, muonPhi, muonMass;

      for (std::int64_t entryId = 0; entryId < recordBatch->num_rows(); ++entryId) {
        if (nMuons[entryId] != 2)
          continue;

        fill_vector_from_arrow(entryId, *muonChargeArr, muonCharge);

        if (muonCharge[0] == muonCharge[1]) {
          continue;
        }

        fill_vector_from_arrow(entryId, *muonPtArr, muonPt);
        fill_vector_from_arrow(entryId, *muonEtaArr, muonEta);
        fill_vector_from_arrow(entryId, *muonPhiArr, muonPhi);
        fill_vector_from_arrow(entryId, *muonMassArr, muonMass);

        float x_sum = 0.;
        float y_sum = 0.;
        float z_sum = 0.;
        float e_sum = 0.;
        for (std::size_t i = 0u; i < 2; ++i) {
          // Convert to (e, x, y, z) coordinate system and update sums
          const auto x = muonPt[i] * std::cos(muonPhi[i]);
          x_sum += x;
          const auto y = muonPt[i] * std::sin(muonPhi[i]);
          y_sum += y;
          const auto z = muonPt[i] * std::sinh(muonEta[i]);
          z_sum += z;
          const auto e =
              std::sqrt(x * x + y * y + z * z + muonMass[i] * muonMass[i]);
          e_sum += e;
        }
        // Return invariant mass with (+, -, -, -) metric
        auto fmass = std::sqrt(e_sum * e_sum - x_sum * x_sum - y_sum * y_sum -
                               z_sum * z_sum);
        hMass->Fill(fmass);
      }
  }

  auto ts_end = std::chrono::steady_clock::now();
  auto runtime_init =
      std::chrono::duration_cast<std::chrono::microseconds>(ts_first - ts_init)
          .count();
  auto runtime_analyze =
      std::chrono::duration_cast<std::chrono::microseconds>(ts_end - ts_first)
          .count();

  if (!histo_path.empty())
    save_histogram(hMass.get(), histo_path);

  return std::make_pair(runtime_init, runtime_analyze);
}

static AnalysisTime_t analysis_parquet(const std::string &path,
                                       const std::string &histo_path) {
  auto ts_init = std::chrono::steady_clock::now();

  auto hMass =
      std::make_unique<TH1D>("Dimuon_mass", "Dimuon_mass", 2000, 0.25, 300);

  arrow::Status st;
  arrow::MemoryPool *pool = arrow::default_memory_pool();

  parquet::arrow::FileReaderBuilder reader_builder;
  st = reader_builder.OpenFile(path);
  if (!st.ok()) {
    throw std::runtime_error("could not create reader builder");
  }
  reader_builder.memory_pool(pool);

  auto reader = reader_builder.Build().ValueOrDie();
  reader->set_use_threads(false);
  auto n_row_groups = reader->num_row_groups();
  std::shared_ptr<arrow::Table> table;

  std::shared_ptr<arrow::Schema> schema;
  st = reader->GetSchema(&schema);
  if (!st.ok()) {
    throw std::runtime_error("could not get schema");
  }

  std::vector<std::int32_t> columns;
  for (const auto colName : columnNames) {
    columns.emplace_back(schema->GetFieldIndex(colName));
  }

  std::chrono::steady_clock::time_point ts_first =
      std::chrono::steady_clock::now();

  for (std::int32_t row_group = 0; row_group < n_row_groups; ++row_group) {
    PARQUET_ASSIGN_OR_THROW(auto table, reader->ReadRowGroup(row_group, columns));

    assert(table->GetColumnByName("nMuon")->num_chunks() == 1);
    assert(table->GetColumnByName("Muon_charge")->num_chunks() == 1);
    assert(table->GetColumnByName("Muon_pt")->num_chunks() == 1);
    assert(table->GetColumnByName("Muon_eta")->num_chunks() == 1);
    assert(table->GetColumnByName("Muon_phi")->num_chunks() == 1);
    assert(table->GetColumnByName("Muon_mass")->num_chunks() == 1);

    auto nMuonArr = std::static_pointer_cast<arrow::Int32Array>(
        table->GetColumnByName("nMuon")->chunk(0));
    auto muonChargeArr = std::static_pointer_cast<arrow::ListArray>(
        table->GetColumnByName("Muon_charge")->chunk(0));
    auto muonPtArr = std::static_pointer_cast<arrow::ListArray>(
        table->GetColumnByName("Muon_pt")->chunk(0));
    auto muonEtaArr = std::static_pointer_cast<arrow::ListArray>(
        table->GetColumnByName("Muon_eta")->chunk(0));
    auto muonPhiArr = std::static_pointer_cast<arrow::ListArray>(
        table->GetColumnByName("Muon_phi")->chunk(0));
    auto muonMassArr = std::static_pointer_cast<arrow::ListArray>(
        table->GetColumnByName("Muon_mass")->chunk(0));

    auto rawNMuonVals = nMuonArr->raw_values();
    ROOT::RVec<std::int32_t> nMuons(rawNMuonVals,
                                    rawNMuonVals + nMuonArr->length());

    ROOT::RVec<std::int32_t> muonCharge;
    ROOT::RVec<float> muonPt, muonEta, muonPhi, muonMass;

    for (std::int64_t entryId = 0; entryId < table->num_rows(); ++entryId) {
      if (nMuons[entryId] != 2)
        continue;

      fill_vector_from_arrow(entryId, *muonChargeArr, muonCharge);

      if (muonCharge[0] == muonCharge[1]) {
        continue;
      }

      fill_vector_from_arrow(entryId, *muonPtArr, muonPt);
      fill_vector_from_arrow(entryId, *muonEtaArr, muonEta);
      fill_vector_from_arrow(entryId, *muonPhiArr, muonPhi);
      fill_vector_from_arrow(entryId, *muonMassArr, muonMass);

      float x_sum = 0.;
      float y_sum = 0.;
      float z_sum = 0.;
      float e_sum = 0.;
      for (std::size_t i = 0u; i < 2; ++i) {
        // Convert to (e, x, y, z) coordinate system and update sums
        const auto x = muonPt[i] * std::cos(muonPhi[i]);
        x_sum += x;
        const auto y = muonPt[i] * std::sin(muonPhi[i]);
        y_sum += y;
        const auto z = muonPt[i] * std::sinh(muonEta[i]);
        z_sum += z;
        const auto e =
            std::sqrt(x * x + y * y + z * z + muonMass[i] * muonMass[i]);
        e_sum += e;
      }
      // Return invariant mass with (+, -, -, -) metric
      auto fmass = std::sqrt(e_sum * e_sum - x_sum * x_sum - y_sum * y_sum -
                             z_sum * z_sum);
      hMass->Fill(fmass);
    }
  }

  auto ts_end = std::chrono::steady_clock::now();
  auto runtime_init =
      std::chrono::duration_cast<std::chrono::microseconds>(ts_first - ts_init)
          .count();
  auto runtime_analyze =
      std::chrono::duration_cast<std::chrono::microseconds>(ts_end - ts_first)
          .count();

  if (!histo_path.empty())
    save_histogram(hMass.get(), histo_path);

  return std::make_pair(runtime_init, runtime_analyze);
}

static AnalysisTime_t analysis_rntuple(std::string_view ntuple_path,
                                       const std::string &histo_path) {
  auto ts_init = std::chrono::steady_clock::now();

  auto ntuple = ROOT::RNTupleReader::Open("Events", ntuple_path);

  auto hMass =
      std::make_unique<TH1D>("Dimuon_mass", "Dimuon_mass", 2000, 0.25, 300);

  auto viewNMuon = ntuple->GetView<std::int32_t>("nMuon");
  auto viewMuonCharge =
      ntuple->GetView<ROOT::RVec<std::int32_t>>("Muon_charge");
  auto viewMuonPt = ntuple->GetView<ROOT::RVec<float>>("Muon_pt");
  auto viewMuonEta = ntuple->GetView<ROOT::RVec<float>>("Muon_eta");
  auto viewMuonPhi = ntuple->GetView<ROOT::RVec<float>>("Muon_phi");
  auto viewMuonMass = ntuple->GetView<ROOT::RVec<float>>("Muon_mass");

  std::chrono::steady_clock::time_point ts_first =
      std::chrono::steady_clock::now();

  for (auto entryId : ntuple->GetEntryRange()) {
    if (viewNMuon(entryId) != 2)
      continue;

    auto charges = viewMuonCharge(entryId);
    if (charges[0] == charges[1])
      continue;

    auto pt = viewMuonPt(entryId);
    auto eta = viewMuonEta(entryId);
    auto phi = viewMuonPhi(entryId);
    auto mass = viewMuonMass(entryId);

    float x_sum = 0.;
    float y_sum = 0.;
    float z_sum = 0.;
    float e_sum = 0.;
    for (std::size_t i = 0u; i < 2; ++i) {
      // Convert to (e, x, y, z) coordinate system and update sums
      const auto x = pt[i] * std::cos(phi[i]);
      x_sum += x;
      const auto y = pt[i] * std::sin(phi[i]);
      y_sum += y;
      const auto z = pt[i] * std::sinh(eta[i]);
      z_sum += z;
      const auto e = std::sqrt(x * x + y * y + z * z + mass[i] * mass[i]);
      e_sum += e;
    }
    // Return invariant mass with (+, -, -, -) metric
    auto fmass = std::sqrt(e_sum * e_sum - x_sum * x_sum - y_sum * y_sum -
                           z_sum * z_sum);
    hMass->Fill(fmass);
  }

  auto ts_end = std::chrono::steady_clock::now();
  auto runtime_init =
      std::chrono::duration_cast<std::chrono::microseconds>(ts_first - ts_init)
          .count();
  auto runtime_analyze =
      std::chrono::duration_cast<std::chrono::microseconds>(ts_end - ts_first)
          .count();

  if (!histo_path.empty())
    save_histogram(hMass.get(), histo_path);

  return std::make_pair(runtime_init, runtime_analyze);
}

static AnalysisTime_t analysis_rdf(ROOT::RDataFrame &df,
                                   const std::string &histo_path) {
  auto ts_init = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point ts_first;
  bool ts_first_set = false;

  auto df_timing = df.Define("TIMING", [&ts_first, &ts_first_set]() {
                       if (!ts_first_set)
                         ts_first = std::chrono::steady_clock::now();
                       ts_first_set = true;
                       return ts_first_set;
                     }).Filter([](bool b) { return b; }, {"TIMING"});
  auto df_2mu =
      df_timing.Filter([](unsigned int s) { return s == 2; }, {"nMuon"});
  auto df_os = df_2mu.Filter(
      [](const ROOT::VecOps::RVec<int> &c) { return c[0] != c[1]; },
      {"Muon_charge"});
  auto df_mass = df_os.Define("Dimuon_mass", ROOT::VecOps::InvariantMass<float>,
                              {"Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass"});
  auto hMass = df_mass.Histo1D<float>(
      {"Dimuon_mass", "Dimuon_mass", 2000, 0.25, 300}, "Dimuon_mass");

  *hMass;
  auto ts_end = std::chrono::steady_clock::now();
  auto runtime_init =
      std::chrono::duration_cast<std::chrono::microseconds>(ts_first - ts_init)
          .count();
  auto runtime_analyze =
      std::chrono::duration_cast<std::chrono::microseconds>(ts_end - ts_first)
          .count();

  if (!histo_path.empty())
    save_histogram(hMass.GetPtr(), histo_path);

  return std::make_pair(runtime_init, runtime_analyze);
}

static void Usage(const char *progname) {
  printf("%s INPUT_PATH [HISTO_PATH]\n", progname);
}

int main(int argc, char **argv) {
  auto ts_init = std::chrono::steady_clock::now();

  if (argc < 2) {
    Usage(argv[0]);
    return 1;
  }

  std::string input_path = argv[1];
  std::string basename, suffix;
  split_path(input_path, &basename, &suffix);
  auto fmt = get_file_format(suffix);

  std::string histo_path{};
  if (argc >= 3)
    histo_path = argv[2];

  AnalysisTime_t runtime_analysis;
  switch (fmt) {
  case FileFormat::rntuple: {
    runtime_analysis = analysis_rntuple(input_path, histo_path);
  } break;
  case FileFormat::parquet: {
    runtime_analysis = analysis_parquet(input_path, histo_path);
    break;
  }
  case FileFormat::orc: {
    runtime_analysis = analysis_orc(input_path, histo_path);
    break;
  }
  default:
    std::cerr << "Invalid file format: " << suffix << std::endl;
    return 1;
  }

  auto ts_end = std::chrono::steady_clock::now();
  auto runtime_main =
      std::chrono::duration_cast<std::chrono::microseconds>(ts_end - ts_init)
          .count();

  std::cout << runtime_analysis.first << ", " << runtime_analysis.second << ", "
            << runtime_main << std::endl;

  return 0;
}
