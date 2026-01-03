#pragma once

class Context;
struct XpbdData;

class XpbdGpuProfiler
{
public:
	XpbdGpuProfiler(Context& context, XpbdData& simData);
	XpbdGpuProfiler(const XpbdGpuProfiler& rhs) = delete;
	XpbdGpuProfiler(XpbdGpuProfiler&& rhs) = delete;
	XpbdGpuProfiler& operator=(const XpbdGpuProfiler& rhs) = delete;
	XpbdGpuProfiler& operator=(XpbdGpuProfiler&& rhs) = delete;
	~XpbdGpuProfiler();
	Context& context_;

	void CalculateGpuTime(XpbdData& simData);

	vk::raii::QueryPool timestamp_pool_{ nullptr };
	uint32_t timestamp_steps_ = 0;

	uint32_t slots_pre_hashing = 6;
	uint32_t slots_spatial_hashing_ = 8;
	uint32_t slots_per_iteration_ = 18;
	uint32_t slots_post_iteration_ = 6;
	uint32_t slots_calculate_normals_ = 4;

	uint32_t slots_per_compute_ = 0;

	float pass_total_time_ = 0.0f;

	std::array<std::string, 21> labels_ = { "Wind", "Intergrate", "ClearLambdas", "HashBuild", "RadixSort", "BuildCell", "BuildNeighbor", "SolveStretch", "SolveShear", "SolveBend", "SolveArea", "SolveSoftbodyStretch", "SolveSoftbodyVolume","SolveSelfCollision", "SolveInterCollision", "ApplyDeltas", "SolveLRA", "CollideSdf", "Update", "CalculateNormals", "Total" };
	std::unordered_map<std::string, double> label_time_;
	std::unordered_map<std::string, double> label_avg_time_;
	uint32_t time_count_ = 0;

private:
	void CreateQueryPool();

};