#pragma once

class Context;
struct ClothData;

class ClothGpuProfiler
{
public:
	ClothGpuProfiler(Context& context, ClothData& simData);
	ClothGpuProfiler(const ClothGpuProfiler& rhs) = delete;
	ClothGpuProfiler(ClothGpuProfiler&& rhs) = delete;
	ClothGpuProfiler& operator=(const ClothGpuProfiler& rhs) = delete;
	ClothGpuProfiler& operator=(ClothGpuProfiler&& rhs) = delete;
	~ClothGpuProfiler();
	Context& context_;

	void CalculateGpuTime(ClothData& simData);

	vk::raii::QueryPool timestamp_pool_{ nullptr };
	uint32_t timestamp_steps_ = 0;

	uint32_t slots_pre_hashing = 6;
	uint32_t slots_spatial_hashing_ = 8;
	uint32_t slots_per_iteration_ = 12;
	uint32_t slots_post_iteration_ = 6;
	uint32_t slots_calculate_normals_ = 4;

	uint32_t slots_per_compute_ = 0;

	float pass_total_time_ = 0.0f;

	std::array<std::string, 18> labels_ = { 
		"Wind", "Intergrate", "ClearLambdas", 
		"HashBuild", "RadixSort", "BuildCell", "BuildNeighbor", 
		"SolveStretch", "SolveShear", "SolveBend", "SolveArea","SolveSelfCollision", "ApplyDeltas", 
		"SolveLRA", "CollideSdf", "Update", "CalculateNormals", "Total" };
	std::unordered_map<std::string, double> label_time_;
	std::unordered_map<std::string, double> label_avg_time_;
	uint32_t time_count_ = 0;

private:
	void CreateQueryPool();

};