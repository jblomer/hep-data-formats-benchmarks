#include <ROOT/RArrowDS.hxx>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RNTupleReader.hxx>

#include <TCanvas.h>
#include <TH1D.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TSystem.h>

#include <arrow/adapters/orc/adapter.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <chrono>
#include <iostream>
#include <string>

#include "util.hxx"

constexpr double kKaonMassMeV = 493.677;

static double GetP2(double px, double py, double pz) {
  return px * px + py * py + pz * pz;
}

static double GetKE(double px, double py, double pz) {
  double p2 = GetP2(px, py, pz);
  return sqrt(p2 + kKaonMassMeV * kKaonMassMeV);
}

const std::vector<std::string> columnNames = {
    "H1_PX",
    "H1_PY",
    "H1_PZ",
    "H1_ProbK",
    "H1_ProbPi",
    "H1_isMuon",
    "H2_PX",
    "H2_PY",
    "H2_PZ",
    "H2_ProbK",
    "H2_ProbPi",
    "H2_isMuon",
    "H3_PX",
    "H3_PY",
    "H3_PZ",
    "H3_ProbK",
    "H3_ProbPi",
    "H3_isMuon",
};

static AnalysisTime_t analysis_orc(const std::string &path,
                                   const std::string &histo_path) {
  auto ts_init = std::chrono::steady_clock::now();
  auto hMass = std::make_unique<TH1D>("B_mass", "", 500, 5050, 5500);

  arrow::MemoryPool *pool = arrow::default_memory_pool();
  auto localFile = arrow::io::ReadableFile::Open(path, pool).ValueOrDie();
  auto reader = arrow::adapters::orc::ORCFileReader::Open(localFile, pool).ValueOrDie();

  auto schema = reader->ReadSchema().ValueOrDie();
  auto nStripes = reader->NumberOfStripes();
  std::shared_ptr<arrow::RecordBatch> recordBatch;

  std::chrono::steady_clock::time_point ts_first =
      std::chrono::steady_clock::now();

  for (std::int32_t stripe = 0; stripe < nStripes; ++stripe) {
    recordBatch = reader->ReadStripe(stripe, columnNames).ValueOrDie();

    auto arrH1IsMuon = std::static_pointer_cast<arrow::Int32Array>(
        recordBatch->GetColumnByName("H1_isMuon"));
    auto arrH2IsMuon = std::static_pointer_cast<arrow::Int32Array>(
        recordBatch->GetColumnByName("H2_isMuon"));
    auto arrH3IsMuon = std::static_pointer_cast<arrow::Int32Array>(
        recordBatch->GetColumnByName("H3_isMuon"));

    auto arrH1PX = std::static_pointer_cast<arrow::DoubleArray>(
        recordBatch->GetColumnByName("H1_PX"));
    auto arrH1PY = std::static_pointer_cast<arrow::DoubleArray>(
        recordBatch->GetColumnByName("H1_PY"));
    auto arrH1PZ = std::static_pointer_cast<arrow::DoubleArray>(
        recordBatch->GetColumnByName("H1_PZ"));
    auto arrH1ProbK = std::static_pointer_cast<arrow::DoubleArray>(
        recordBatch->GetColumnByName("H1_ProbK"));
    auto arrH1ProbPi = std::static_pointer_cast<arrow::DoubleArray>(
        recordBatch->GetColumnByName("H1_ProbPi"));

    auto arrH2PX = std::static_pointer_cast<arrow::DoubleArray>(
        recordBatch->GetColumnByName("H2_PX"));
    auto arrH2PY = std::static_pointer_cast<arrow::DoubleArray>(
        recordBatch->GetColumnByName("H2_PY"));
    auto arrH2PZ = std::static_pointer_cast<arrow::DoubleArray>(
        recordBatch->GetColumnByName("H2_PZ"));
    auto arrH2ProbK = std::static_pointer_cast<arrow::DoubleArray>(
        recordBatch->GetColumnByName("H2_ProbK"));
    auto arrH2ProbPi = std::static_pointer_cast<arrow::DoubleArray>(
        recordBatch->GetColumnByName("H2_ProbPi"));

    auto arrH3PX = std::static_pointer_cast<arrow::DoubleArray>(
        recordBatch->GetColumnByName("H3_PX"));
    auto arrH3PY = std::static_pointer_cast<arrow::DoubleArray>(
        recordBatch->GetColumnByName("H3_PY"));
    auto arrH3PZ = std::static_pointer_cast<arrow::DoubleArray>(
        recordBatch->GetColumnByName("H3_PZ"));
    auto arrH3ProbK = std::static_pointer_cast<arrow::DoubleArray>(
        recordBatch->GetColumnByName("H3_ProbK"));
    auto arrH3ProbPi = std::static_pointer_cast<arrow::DoubleArray>(
        recordBatch->GetColumnByName("H3_ProbPi"));

    auto rawH1IsMuon = arrH1IsMuon->raw_values();
    ROOT::RVec<std::int32_t> valsH1IsMuon(rawH1IsMuon,
                                          rawH1IsMuon + arrH1IsMuon->length());
    auto rawH2IsMuon = arrH2IsMuon->raw_values();
    ROOT::RVec<std::int32_t> valsH2IsMuon(rawH2IsMuon,
                                          rawH2IsMuon + arrH2IsMuon->length());
    auto rawH3IsMuon = arrH3IsMuon->raw_values();
    ROOT::RVec<std::int32_t> valsH3IsMuon(rawH3IsMuon,
                                          rawH3IsMuon + arrH3IsMuon->length());

    auto rawH1PX = arrH1PX->raw_values();
    ROOT::RVec<double> valsH1PX(rawH1PX, rawH1PX + arrH1PX->length());
    auto rawH1PY = arrH1PY->raw_values();
    ROOT::RVec<double> valsH1PY(rawH1PY, rawH1PY + arrH1PY->length());
    auto rawH1PZ = arrH1PZ->raw_values();
    ROOT::RVec<double> valsH1PZ(rawH1PZ, rawH1PZ + arrH1PZ->length());
    auto rawH1ProbK = arrH1ProbK->raw_values();
    ROOT::RVec<double> valsH1ProbK(rawH1ProbK,
                                         rawH1ProbK + arrH1ProbK->length());
    auto rawH1ProbPi = arrH1ProbPi->raw_values();
    ROOT::RVec<double> valsH1ProbPi(rawH1ProbPi,
                                          rawH1ProbPi + arrH1ProbPi->length());

    auto rawH2PX = arrH2PX->raw_values();
    ROOT::RVec<double> valsH2PX(rawH2PX, rawH2PX + arrH2PX->length());
    auto rawH2PY = arrH2PY->raw_values();
    ROOT::RVec<double> valsH2PY(rawH2PY, rawH2PY + arrH2PY->length());
    auto rawH2PZ = arrH2PZ->raw_values();
    ROOT::RVec<double> valsH2PZ(rawH2PZ, rawH2PZ + arrH2PZ->length());
    auto rawH2ProbK = arrH2ProbK->raw_values();
    ROOT::RVec<double> valsH2ProbK(rawH2ProbK,
                                         rawH2ProbK + arrH2ProbK->length());
    auto rawH2ProbPi = arrH2ProbPi->raw_values();
    ROOT::RVec<double> valsH2ProbPi(rawH2ProbPi,
                                          rawH2ProbPi + arrH2ProbPi->length());

    auto rawH3PX = arrH3PX->raw_values();
    ROOT::RVec<double> valsH3PX(rawH3PX, rawH3PX + arrH3PX->length());
    auto rawH3PY = arrH3PY->raw_values();
    ROOT::RVec<double> valsH3PY(rawH3PY, rawH3PY + arrH3PY->length());
    auto rawH3PZ = arrH3PZ->raw_values();
    ROOT::RVec<double> valsH3PZ(rawH3PZ, rawH3PZ + arrH3PZ->length());
    auto rawH3ProbK = arrH3ProbK->raw_values();
    ROOT::RVec<double> valsH3ProbK(rawH3ProbK,
                                         rawH3ProbK + arrH3ProbK->length());
    auto rawH3ProbPi = arrH3ProbPi->raw_values();
    ROOT::RVec<double> valsH3ProbPi(rawH3ProbPi,
                                          rawH3ProbPi + arrH3ProbPi->length());

    for (std::int64_t entryId = 0; entryId < recordBatch->num_rows(); ++entryId) {
      if (valsH1IsMuon[entryId] || valsH2IsMuon[entryId] ||
          valsH3IsMuon[entryId]) {
        continue;
      }

      constexpr double prob_k_cut = 0.5;
      if (valsH1ProbK[entryId] < prob_k_cut)
        continue;
      if (valsH2ProbK[entryId] < prob_k_cut)
        continue;
      if (valsH3ProbK[entryId] < prob_k_cut)
        continue;

      constexpr double prob_pi_cut = 0.5;
      if (valsH1ProbPi[entryId] > prob_pi_cut)
        continue;
      if (valsH2ProbPi[entryId] > prob_pi_cut)
        continue;
      if (valsH3ProbPi[entryId] > prob_pi_cut)
        continue;

      double b_px = valsH1PX[entryId] + valsH2PX[entryId] + valsH3PX[entryId];
      double b_py = valsH1PY[entryId] + valsH2PY[entryId] + valsH3PY[entryId];
      double b_pz = valsH1PZ[entryId] + valsH2PZ[entryId] + valsH3PZ[entryId];
      double b_p2 = GetP2(b_px, b_py, b_pz);
      double k1_E =
      GetKE(valsH1PX[entryId], valsH1PY[entryId], valsH1PZ[entryId]);
      double k2_E =
      GetKE(valsH2PX[entryId], valsH2PY[entryId], valsH2PZ[entryId]);
      double k3_E =
      GetKE(valsH3PX[entryId], valsH3PY[entryId], valsH3PZ[entryId]);
      double b_E = k1_E + k2_E + k3_E;
      double b_mass = sqrt(b_E * b_E - b_p2);
      hMass->Fill(b_mass);
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
  auto hMass = std::make_unique<TH1D>("B_mass", "", 500, 5050, 5500);

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

    auto arrH1IsMuon = std::static_pointer_cast<arrow::Int32Array>(
        table->GetColumnByName("H1_isMuon")->chunk(0));
    auto arrH2IsMuon = std::static_pointer_cast<arrow::Int32Array>(
        table->GetColumnByName("H2_isMuon")->chunk(0));
    auto arrH3IsMuon = std::static_pointer_cast<arrow::Int32Array>(
        table->GetColumnByName("H3_isMuon")->chunk(0));

    auto arrH1PX = std::static_pointer_cast<arrow::DoubleArray>(
        table->GetColumnByName("H1_PX")->chunk(0));
    auto arrH1PY = std::static_pointer_cast<arrow::DoubleArray>(
        table->GetColumnByName("H1_PY")->chunk(0));
    auto arrH1PZ = std::static_pointer_cast<arrow::DoubleArray>(
        table->GetColumnByName("H1_PZ")->chunk(0));
    auto arrH1ProbK = std::static_pointer_cast<arrow::DoubleArray>(
        table->GetColumnByName("H1_ProbK")->chunk(0));
    auto arrH1ProbPi = std::static_pointer_cast<arrow::DoubleArray>(
        table->GetColumnByName("H1_ProbPi")->chunk(0));

    auto arrH2PX = std::static_pointer_cast<arrow::DoubleArray>(
        table->GetColumnByName("H2_PX")->chunk(0));
    auto arrH2PY = std::static_pointer_cast<arrow::DoubleArray>(
        table->GetColumnByName("H2_PY")->chunk(0));
    auto arrH2PZ = std::static_pointer_cast<arrow::DoubleArray>(
        table->GetColumnByName("H2_PZ")->chunk(0));
    auto arrH2ProbK = std::static_pointer_cast<arrow::DoubleArray>(
        table->GetColumnByName("H2_ProbK")->chunk(0));
    auto arrH2ProbPi = std::static_pointer_cast<arrow::DoubleArray>(
        table->GetColumnByName("H2_ProbPi")->chunk(0));

    auto arrH3PX = std::static_pointer_cast<arrow::DoubleArray>(
        table->GetColumnByName("H3_PX")->chunk(0));
    auto arrH3PY = std::static_pointer_cast<arrow::DoubleArray>(
        table->GetColumnByName("H3_PY")->chunk(0));
    auto arrH3PZ = std::static_pointer_cast<arrow::DoubleArray>(
        table->GetColumnByName("H3_PZ")->chunk(0));
    auto arrH3ProbK = std::static_pointer_cast<arrow::DoubleArray>(
        table->GetColumnByName("H3_ProbK")->chunk(0));
    auto arrH3ProbPi = std::static_pointer_cast<arrow::DoubleArray>(
        table->GetColumnByName("H3_ProbPi")->chunk(0));

    auto rawH1IsMuon = arrH1IsMuon->raw_values();
    ROOT::RVec<std::int32_t> valsH1IsMuon(rawH1IsMuon,
                                          rawH1IsMuon + arrH1IsMuon->length());
    auto rawH2IsMuon = arrH2IsMuon->raw_values();
    ROOT::RVec<std::int32_t> valsH2IsMuon(rawH2IsMuon,
                                          rawH2IsMuon + arrH2IsMuon->length());
    auto rawH3IsMuon = arrH3IsMuon->raw_values();
    ROOT::RVec<std::int32_t> valsH3IsMuon(rawH3IsMuon,
                                          rawH3IsMuon + arrH3IsMuon->length());

    auto rawH1PX = arrH1PX->raw_values();
    ROOT::RVec<double> valsH1PX(rawH1PX, rawH1PX + arrH1PX->length());
    auto rawH1PY = arrH1PY->raw_values();
    ROOT::RVec<double> valsH1PY(rawH1PY, rawH1PY + arrH1PY->length());
    auto rawH1PZ = arrH1PZ->raw_values();
    ROOT::RVec<double> valsH1PZ(rawH1PZ, rawH1PZ + arrH1PZ->length());
    auto rawH1ProbK = arrH1ProbK->raw_values();
    ROOT::RVec<double> valsH1ProbK(rawH1ProbK,
                                         rawH1ProbK + arrH1ProbK->length());
    auto rawH1ProbPi = arrH1ProbPi->raw_values();
    ROOT::RVec<double> valsH1ProbPi(rawH1ProbPi,
                                          rawH1ProbPi + arrH1ProbPi->length());

    auto rawH2PX = arrH2PX->raw_values();
    ROOT::RVec<double> valsH2PX(rawH2PX, rawH2PX + arrH2PX->length());
    auto rawH2PY = arrH2PY->raw_values();
    ROOT::RVec<double> valsH2PY(rawH2PY, rawH2PY + arrH2PY->length());
    auto rawH2PZ = arrH2PZ->raw_values();
    ROOT::RVec<double> valsH2PZ(rawH2PZ, rawH2PZ + arrH2PZ->length());
    auto rawH2ProbK = arrH2ProbK->raw_values();
    ROOT::RVec<double> valsH2ProbK(rawH2ProbK,
                                         rawH2ProbK + arrH2ProbK->length());
    auto rawH2ProbPi = arrH2ProbPi->raw_values();
    ROOT::RVec<double> valsH2ProbPi(rawH2ProbPi,
                                          rawH2ProbPi + arrH2ProbPi->length());

    auto rawH3PX = arrH3PX->raw_values();
    ROOT::RVec<double> valsH3PX(rawH3PX, rawH3PX + arrH3PX->length());
    auto rawH3PY = arrH3PY->raw_values();
    ROOT::RVec<double> valsH3PY(rawH3PY, rawH3PY + arrH3PY->length());
    auto rawH3PZ = arrH3PZ->raw_values();
    ROOT::RVec<double> valsH3PZ(rawH3PZ, rawH3PZ + arrH3PZ->length());
    auto rawH3ProbK = arrH3ProbK->raw_values();
    ROOT::RVec<double> valsH3ProbK(rawH3ProbK,
                                         rawH3ProbK + arrH3ProbK->length());
    auto rawH3ProbPi = arrH3ProbPi->raw_values();
    ROOT::RVec<double> valsH3ProbPi(rawH3ProbPi,
                                          rawH3ProbPi + arrH3ProbPi->length());

    for (std::int64_t entryId = 0; entryId < table->num_rows(); ++entryId) {

      if (valsH1IsMuon[entryId] || valsH2IsMuon[entryId] ||
          valsH3IsMuon[entryId]) {
        continue;
      }

      constexpr double prob_k_cut = 0.5;
      if (valsH1ProbK[entryId] < prob_k_cut)
        continue;
      if (valsH2ProbK[entryId] < prob_k_cut)
        continue;
      if (valsH3ProbK[entryId] < prob_k_cut)
        continue;

      constexpr double prob_pi_cut = 0.5;
      if (valsH1ProbPi[entryId] > prob_pi_cut)
        continue;
      if (valsH2ProbPi[entryId] > prob_pi_cut)
        continue;
      if (valsH3ProbPi[entryId] > prob_pi_cut)
        continue;

      double b_px = valsH1PX[entryId] + valsH2PX[entryId] + valsH3PX[entryId];
      double b_py = valsH1PY[entryId] + valsH2PY[entryId] + valsH3PY[entryId];
      double b_pz = valsH1PZ[entryId] + valsH2PZ[entryId] + valsH3PZ[entryId];
      double b_p2 = GetP2(b_px, b_py, b_pz);
      double k1_E =
      GetKE(valsH1PX[entryId], valsH1PY[entryId], valsH1PZ[entryId]);
      double k2_E =
      GetKE(valsH2PX[entryId], valsH2PY[entryId], valsH2PZ[entryId]);
      double k3_E =
      GetKE(valsH3PX[entryId], valsH3PY[entryId], valsH3PZ[entryId]);
      double b_E = k1_E + k2_E + k3_E;
      double b_mass = sqrt(b_E * b_E - b_p2);
      hMass->Fill(b_mass);
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

static AnalysisTime_t analysis_rntuple(const std::string &path,
                                       const std::string &histo_path) {
  auto ts_init = std::chrono::steady_clock::now();
  auto hMass = std::make_unique<TH1D>("B_mass", "", 500, 5050, 5500);

  auto ntuple = ROOT::RNTupleReader::Open("DecayTree", path);

  auto viewH1IsMuon = ntuple->GetDirectAccessView<int>("H1_isMuon");
  auto viewH2IsMuon = ntuple->GetDirectAccessView<int>("H2_isMuon");
  auto viewH3IsMuon = ntuple->GetDirectAccessView<int>("H3_isMuon");

  auto viewH1PX = ntuple->GetDirectAccessView<double>("H1_PX");
  auto viewH1PY = ntuple->GetDirectAccessView<double>("H1_PY");
  auto viewH1PZ = ntuple->GetDirectAccessView<double>("H1_PZ");
  auto viewH1ProbK = ntuple->GetDirectAccessView<double>("H1_ProbK");
  auto viewH1ProbPi = ntuple->GetDirectAccessView<double>("H1_ProbPi");

  auto viewH2PX = ntuple->GetDirectAccessView<double>("H2_PX");
  auto viewH2PY = ntuple->GetDirectAccessView<double>("H2_PY");
  auto viewH2PZ = ntuple->GetDirectAccessView<double>("H2_PZ");
  auto viewH2ProbK = ntuple->GetDirectAccessView<double>("H2_ProbK");
  auto viewH2ProbPi = ntuple->GetDirectAccessView<double>("H2_ProbPi");

  auto viewH3PX = ntuple->GetDirectAccessView<double>("H3_PX");
  auto viewH3PY = ntuple->GetDirectAccessView<double>("H3_PY");
  auto viewH3PZ = ntuple->GetDirectAccessView<double>("H3_PZ");
  auto viewH3ProbK = ntuple->GetDirectAccessView<double>("H3_ProbK");
  auto viewH3ProbPi = ntuple->GetDirectAccessView<double>("H3_ProbPi");

  unsigned nevents = 0;
  std::chrono::steady_clock::time_point ts_first =
      std::chrono::steady_clock::now();
  for (auto i : ntuple->GetEntryRange()) {
    if (viewH1IsMuon(i) || viewH2IsMuon(i) || viewH3IsMuon(i)) {
      continue;
    }

    constexpr double prob_k_cut = 0.5;
    if (viewH1ProbK(i) < prob_k_cut)
      continue;
    if (viewH2ProbK(i) < prob_k_cut)
      continue;
    if (viewH3ProbK(i) < prob_k_cut)
      continue;

    constexpr double prob_pi_cut = 0.5;
    if (viewH1ProbPi(i) > prob_pi_cut)
      continue;
    if (viewH2ProbPi(i) > prob_pi_cut)
      continue;
    if (viewH3ProbPi(i) > prob_pi_cut)
      continue;

    double b_px = viewH1PX(i) + viewH2PX(i) + viewH3PX(i);
    double b_py = viewH1PY(i) + viewH2PY(i) + viewH3PY(i);
    double b_pz = viewH1PZ(i) + viewH2PZ(i) + viewH3PZ(i);
    double b_p2 = GetP2(b_px, b_py, b_pz);
    double k1_E = GetKE(viewH1PX(i), viewH1PY(i), viewH1PZ(i));
    double k2_E = GetKE(viewH2PX(i), viewH2PY(i), viewH2PZ(i));
    double k3_E = GetKE(viewH3PX(i), viewH3PY(i), viewH3PZ(i));
    double b_E = k1_E + k2_E + k3_E;
    double b_mass = sqrt(b_E * b_E - b_p2);
    hMass->Fill(b_mass);
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

static AnalysisTime_t analysis_rdf(ROOT::RDataFrame &frame,
                                   const std::string &histo_path) {
  auto ts_init = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point ts_first;

  auto fn_muon_cut_and_stopwatch = [&](unsigned int slot, ULong64_t entry,
                                       int is_muon) {
    if (entry == 0) {
      ts_first = std::chrono::steady_clock::now();
    }
    return !is_muon;
  };
  auto fn_muon_cut = [](int is_muon) { return !is_muon; };
  auto fn_k_cut = [](double prob_k) { return prob_k > 0.5; };
  auto fn_pi_cut = [](double prob_pi) { return prob_pi < 0.5; };
  auto fn_sum = [](double p1, double p2, double p3) { return p1 + p2 + p3; };
  auto fn_mass = [](double B_E, double B_P2) {
    double r = sqrt(B_E * B_E - B_P2);
    return r;
  };

  auto df_muon_cut = frame
                         .Filter(fn_muon_cut_and_stopwatch,
                                 {"rdfslot_", "rdfentry_", "H1_isMuon"})
                         .Filter(fn_muon_cut, {"H2_isMuon"})
                         .Filter(fn_muon_cut, {"H3_isMuon"});
  auto df_k_cut = df_muon_cut.Filter(fn_k_cut, {"H1_ProbK"})
                      .Filter(fn_k_cut, {"H2_ProbK"})
                      .Filter(fn_k_cut, {"H3_ProbK"});
  auto df_pi_cut = df_k_cut.Filter(fn_pi_cut, {"H1_ProbPi"})
                       .Filter(fn_pi_cut, {"H2_ProbPi"})
                       .Filter(fn_pi_cut, {"H3_ProbPi"});
  auto df_mass = df_pi_cut.Define("B_PX", fn_sum, {"H1_PX", "H2_PX", "H3_PX"})
                     .Define("B_PY", fn_sum, {"H1_PY", "H2_PY", "H3_PY"})
                     .Define("B_PZ", fn_sum, {"H1_PZ", "H2_PZ", "H3_PZ"})
                     .Define("B_P2", GetP2, {"B_PX", "B_PY", "B_PZ"})
                     .Define("K1_E", GetKE, {"H1_PX", "H1_PY", "H1_PZ"})
                     .Define("K2_E", GetKE, {"H2_PX", "H2_PY", "H2_PZ"})
                     .Define("K3_E", GetKE, {"H3_PX", "H3_PY", "H3_PZ"})
                     .Define("B_E", fn_sum, {"K1_E", "K2_E", "K3_E"})
                     .Define("B_m", fn_mass, {"B_E", "B_P2"});
  auto hMass = df_mass.Histo1D<double>({"B_mass", "", 500, 5050, 5500}, "B_m");

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
  case FileFormat::orc: {
    runtime_analysis = analysis_orc(input_path, histo_path);
  } break;
  case FileFormat::parquet: {
    runtime_analysis = analysis_parquet(input_path, histo_path);
  } break;
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
