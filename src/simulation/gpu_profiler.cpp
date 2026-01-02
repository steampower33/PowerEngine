#include "context.h"
#include "sim_data.h"

#include "gpu_profiler.h"

GpuProfiler::GpuProfiler(Context& context, SimData& simData)
	: context_(context)
{
	CreateQueryPool();

	slots_per_compute_ =
		1
		+ simData.substeps *
		(slots_pre_hashing + slots_spatial_hashing_
			+ simData.iterations * slots_per_iteration_
			+ slots_post_iteration_)
		+ slots_calculate_normals_
		+ 1;
}

GpuProfiler::~GpuProfiler()
{

}

void GpuProfiler::CreateQueryPool() {
	vk::QueryPoolCreateInfo queryInfo = {};
	queryInfo.queryType = vk::QueryType::eTimestamp;
	queryInfo.queryCount = 2048;

	timestamp_pool_ = context_.device_.createQueryPool(queryInfo);
}

void GpuProfiler::CalculateGpuTime(SimData& simData, uint32_t broadphase_interval_)
{
	float nsPerTick = context_.physical_device_.getProperties().limits.timestampPeriod;
	float toMs = nsPerTick / 1e6f;

	if (timestamp_steps_ <= 0) return;

	uint32_t numTimestamp = timestamp_steps_;
	std::vector<uint64_t> ts(numTimestamp);

	VkResult res = vkGetQueryPoolResults(
		static_cast<VkDevice>(*context_.device_),
		static_cast<VkQueryPool>(*timestamp_pool_),
		0, numTimestamp,
		ts.size() * sizeof(uint64_t), ts.data(), sizeof(uint64_t),
		VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
	);

	auto delta_ms = [&](uint32_t i0, uint32_t i1) {
		return (ts[i1] - ts[i0]) * toMs;
		};

	float tWind = 0.0f;
	float tIntegrate = 0.0f;
	float tClearLambdas = 0.0f;
	float tHashBuild = 0.0f;
	float tRadixSort = 0.0f;
	float tBuildCell = 0.0f;
	float tBuildNeighbor = 0.0f;
	float tSolveStretch = 0.0f;
	float tSolveShear = 0.0f;
	float tSolveBend = 0.0f;
	float tSolveArea = 0.0f;
	float tSolveSoftbodyStretch = 0.0f;
	float tSolveSoftbodyVolume = 0.0f;
	float tSolveSelfCollision = 0.0f;
	float tSolveInterCollision = 0.0f;
	float tApplyDeltas = 0.0f;
	float tSolveLRA = 0.0f;
	float tCollideSdf = 0.0f;
	float tUpdate = 0.0f;
	float tCalculateNormals = 0.0f;

	uint32_t tsCnt = slots_per_iteration_;
	uint32_t base = 1;
	uint32_t afterIteration = 0;
	for (uint32_t sub = 0; sub < simData.substeps; sub++)
	{
		tWind += delta_ms(base + 0, base + 1);
		tIntegrate += delta_ms(base + 2, base + 3);
		tClearLambdas += delta_ms(base + 4, base + 5);

		if (sub % broadphase_interval_ == 0)
		{
			tHashBuild += delta_ms(base + 6, base + 7);
			tRadixSort += delta_ms(base + 8, base + 9);
			tBuildCell += delta_ms(base + 10, base + 11);
			tBuildNeighbor += delta_ms(base + 12, base + 13);
		}

		uint32_t iterBase = base + 12;
		for (uint32_t it = 0; it < simData.iterations; it++)
		{
			tSolveStretch += delta_ms(iterBase + it * tsCnt + 0, iterBase + it * tsCnt + 1);
			tSolveShear += delta_ms(iterBase + it * tsCnt + 2, iterBase + it * tsCnt + 3);
			tSolveBend += delta_ms(iterBase + it * tsCnt + 4, iterBase + it * tsCnt + 5);
			tSolveArea += delta_ms(iterBase + it * tsCnt + 6, iterBase + it * tsCnt + 7);
			tSolveSoftbodyStretch += delta_ms(iterBase + it * tsCnt + 8, iterBase + it * tsCnt + 9);
			tSolveSoftbodyVolume += delta_ms(iterBase + it * tsCnt + 10, iterBase + it * tsCnt + 11);
			tSolveSelfCollision += delta_ms(iterBase + it * tsCnt + 12, iterBase + it * tsCnt + 13);
			tSolveInterCollision += delta_ms(iterBase + it * tsCnt + 14, iterBase + it * tsCnt + 15);
			tApplyDeltas += delta_ms(iterBase + it * tsCnt + 16, iterBase + it * tsCnt + 17);
		}
		afterIteration = iterBase + simData.iterations * tsCnt;

		tSolveLRA += delta_ms(afterIteration + 0, afterIteration + 1);
		tCollideSdf += delta_ms(afterIteration + 2, afterIteration + 3);
		tUpdate += delta_ms(afterIteration + 4, afterIteration + 5);
	}
	uint32_t afterSubstep = afterIteration + slots_post_iteration_;

	tCalculateNormals = delta_ms(afterSubstep, afterSubstep + 1) + delta_ms(afterSubstep + 2, afterSubstep + 3);

	pass_total_time_ = delta_ms(0, numTimestamp - 1);

	//std::cout << pass_total_time_ << std::endl;

	float total =
		tIntegrate + tClearLambdas +
		tHashBuild + tRadixSort + tBuildCell + tBuildNeighbor +
		tSolveStretch + tSolveSoftbodyStretch + tSolveSoftbodyVolume + tSolveBend + tSolveArea + tSolveSelfCollision + tSolveInterCollision + tApplyDeltas +
		tCollideSdf + tUpdate +
		tCalculateNormals;

	uint32_t c = 0;

	{
		c = 0;
		label_time_[labels_[c++]] = tWind;
		label_time_[labels_[c++]] = tIntegrate;
		label_time_[labels_[c++]] = tClearLambdas;
		label_time_[labels_[c++]] = tHashBuild;
		label_time_[labels_[c++]] = tRadixSort;
		label_time_[labels_[c++]] = tBuildCell;
		label_time_[labels_[c++]] = tBuildNeighbor;
		label_time_[labels_[c++]] = tSolveStretch;
		label_time_[labels_[c++]] = tSolveShear;
		label_time_[labels_[c++]] = tSolveBend;
		label_time_[labels_[c++]] = tSolveArea;
		label_time_[labels_[c++]] = tSolveSoftbodyStretch;
		label_time_[labels_[c++]] = tSolveSoftbodyVolume;
		label_time_[labels_[c++]] = tSolveSelfCollision;
		label_time_[labels_[c++]] = tSolveInterCollision;
		label_time_[labels_[c++]] = tApplyDeltas;
		label_time_[labels_[c++]] = tSolveLRA;
		label_time_[labels_[c++]] = tCollideSdf;
		label_time_[labels_[c++]] = tUpdate;
		label_time_[labels_[c++]] = tCalculateNormals;
		label_time_[labels_[c++]] = total;
	}

	{
		c = 0;
		label_avg_time_[labels_[c++]] += tWind;
		label_avg_time_[labels_[c++]] += tIntegrate;
		label_avg_time_[labels_[c++]] += tClearLambdas;
		label_avg_time_[labels_[c++]] += tHashBuild;
		label_avg_time_[labels_[c++]] += tRadixSort;
		label_avg_time_[labels_[c++]] += tBuildCell;
		label_avg_time_[labels_[c++]] += tBuildNeighbor;
		label_avg_time_[labels_[c++]] += tSolveStretch;
		label_avg_time_[labels_[c++]] += tSolveShear;
		label_avg_time_[labels_[c++]] += tSolveBend;
		label_avg_time_[labels_[c++]] += tSolveArea;
		label_avg_time_[labels_[c++]] += tSolveSoftbodyStretch;
		label_avg_time_[labels_[c++]] += tSolveSoftbodyVolume;
		label_avg_time_[labels_[c++]] += tSolveSelfCollision;
		label_avg_time_[labels_[c++]] += tSolveInterCollision;
		label_avg_time_[labels_[c++]] += tApplyDeltas;
		label_avg_time_[labels_[c++]] += tSolveLRA;
		label_avg_time_[labels_[c++]] += tCollideSdf;
		label_avg_time_[labels_[c++]] += tUpdate;
		label_avg_time_[labels_[c++]] += tCalculateNormals;
		label_avg_time_[labels_[c++]] += total;
	}

	time_count_++;

}