#include <cstdlib>
#include <iostream>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param.h>
#include <zvec/core/interface/index_param_builders.h>

using namespace zvec::core_interface;

constexpr uint32_t kDimension = 5;
const std::string index_name{"test.index"};

Index::Pointer create_index(const BaseIndexParam::Pointer &param,
                            int doc_num = 10) {
  auto index = IndexFactory::CreateAndInitIndex(*param);
  if (!index) {
    std::cout << "Failed to create index." << std::endl;
    return nullptr;
  }

  int ret = index->Open(
      index_name, StorageOptions{StorageOptions::StorageType::kMMAP, true});
  if (ret != 0) {
    std::cout << "Failed to open index." << std::endl;
    return nullptr;
  }

  std::vector<float> vector1{0.418708, 0.809902, 0.823193, 0.598179, 0.0332549};
  VectorData vector_data1;
  vector_data1.vector = DenseVector{vector1.data()};
  index->Add(vector_data1, 0);

  std::vector<float> vector2{0.687774, 0.789588, 0.496138, 0.57487, 0.917617};
  VectorData vector_data2;
  vector_data2.vector = DenseVector{vector2.data()};
  index->Add(vector_data2, 1);

  std::vector<float> vector3{0.333221, 0.962687, 0.467263, 0.448235, 0.475671};
  VectorData vector_data3;
  vector_data3.vector = DenseVector{vector3.data()};
  index->Add(vector_data3, 2);

  std::vector<float> vector4{0.822185, 0.185643, 0.683452, 0.211072, 0.554056};
  VectorData vector_data4;
  vector_data4.vector = DenseVector{vector4.data()};
  index->Add(vector_data4, 3);

  std::vector<float> vector5{0.437057, 0.167281, 0.0770977, 0.428638, 0.241591};
  VectorData vector_data5;
  vector_data5.vector = DenseVector{vector5.data()};
  index->Add(vector_data5, 4);

  std::vector<float> vector6{0.76956, 0.926895, 0.803376, 0.0157961, 0.589042};
  VectorData vector_data6;
  vector_data6.vector = DenseVector{vector6.data()};
  index->Add(vector_data6, 5);

  std::vector<float> vector7{0.493999, 0.641957, 0.761598, 0.94276, 0.425865};
  VectorData vector_data7;
  vector_data7.vector = DenseVector{vector7.data()};
  index->Add(vector_data7, 6);

  std::vector<float> vector8{0.924108, 0.275466, 0.0543329, 0.0731585,
                             0.136344};
  VectorData vector_data8;
  vector_data8.vector = DenseVector{vector8.data()};
  index->Add(vector_data8, 7);

  std::vector<float> vector9{0.186956, 0.69666, 0.0356002, 0.668875, 0.84722};
  VectorData vector_data9;
  vector_data9.vector = DenseVector{vector9.data()};
  index->Add(vector_data9, 8);

  std::vector<float> vector10{0.415294, 0.609278, 0.426765, 0.988832, 0.475556};
  VectorData vector_data10;
  vector_data10.vector = DenseVector{vector10.data()};
  index->Add(vector_data10, 9);

  ret = index->Train();
  if (ret != 0) {
    std::cout << "Failed to train index." << std::endl;
    return nullptr;
  }

  return index;
}

int main() {
  char cmd_buf[100];
  snprintf(cmd_buf, 100, "rm -f %s", index_name.c_str());
  system(cmd_buf);

  auto param = HNSWIndexParamBuilder()
                   .WithMetricType(MetricType::kCosine)
                   .WithDataType(DataType::DT_FP32)
                   .WithQuantizerParam(QuantizerType::kInt8)
                   .WithDimension(kDimension)
                   .WithIsSparse(false)
                   .Build();
  auto index = create_index(param);
  std::cout << "index stats: " << index->GetDocCount() << std::endl;

  // query
  auto query_param = HNSWQueryParamBuilder()
                         .with_topk(10)
                         .with_fetch_vector(true)
                         .with_ef_search(20)
                         .build();

  SearchResult result;
  VectorData query;
  std::vector<float> vector{0.1,0.2,0.13,0.24,-0.5};
  query.vector = DenseVector{vector.data()};
  int ret = index->Search(query, query_param, &result);
  if (ret != 0) {
    std::cout << "Failed to search index." << std::endl;
    return -1;
  }

  std::cout << "query results: " << result.doc_list_.size() << std::endl;
  if (result.doc_list_.size() == 0) {
    std::cout << "No results found." << std::endl;
    return -1;
  }

  for (size_t i = 0; i < result.doc_list_.size(); ++i) {
    std::cout << "[" << i << "] key: " << result.doc_list_[i].key() + 1
              << ", score: " << result.doc_list_[i].score() << std::endl;
  }

  return 0;
}