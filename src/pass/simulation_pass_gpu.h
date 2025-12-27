#pragma once

class Context;
class Swapchain;
class Texture;
class TextureManager;
class Model;
class ModelManager;
class MouseInteractor;
class ParticleManager;

#include "sim_data.h"
#include "sim_ubo.h"

#include "vulkan_utils.h"

#include <vk_radix_sort.h>

class SimulationPassGPU
{
public:
	SimulationPassGPU(Context& context, Swapchain& swapchain, ParticleManager& particleManager, ModelManager& modelManager);
	SimulationPassGPU(const SimulationPassGPU& rhs) = delete;
	SimulationPassGPU(SimulationPassGPU&& rhs) = delete;
	SimulationPassGPU& operator=(const SimulationPassGPU& rhs) = delete;
	SimulationPassGPU& operator=(SimulationPassGPU&& rhs) = delete;
	~SimulationPassGPU();

	Context& context_;
	ParticleManager& particle_manager_;
	ModelManager& model_manager_;

	uint32_t total_particles_;
	uint32_t total_indices_;
	uint32_t total_tri_;

	uint32_t cloth_particles_;
	uint32_t cloth_indices_;
	uint32_t softbody_particles_;
	uint32_t softbody_indices_;

	int broadphase_interval_ = 2;
	int narrowphase_interval_ = 1;

	void UpdateMousePushConstant(Camera& camera, MouseInteractor& mouseInteractor, glm::vec2 viewportSize);
	void UpdateComputeUBO(uint32_t currentFrame, ModelManager& model);

	void RecordCompute(uint32_t currentFrame, vku::TestScene& testScene);

	void ClearCpuTime();
	void CalculateGpuTime();

	void CopySimDatas(const vk::raii::CommandBuffer& cmd);
	void ResetTestScene(const vk::raii::CommandBuffer& cmd, vku::TestScene& testScene);
	void CopyColliders(const vk::raii::CommandBuffer& cmd);

	std::vector<vk::raii::CommandBuffer> cmds_;

	SimData datas_;
	SimUBO ubo_;

	struct SolverConfig {
		bool stretch = true;
		bool shear = true;
		bool bend = false;
		bool area = true;
		bool self_collision = true;
		bool inter_collision = true;

		bool softbody_stretch = true;
		bool softbody_volume = true;
	} solver_config_;

	vk::raii::QueryPool timestamp_pool_{ nullptr };
	uint32_t timestamp_steps_ = 0;
	uint32_t slots_integrate_clear = 4;
	uint32_t slots_spatial_hashing_ = 8;
	uint32_t slots_per_iteration_ = 18;
	uint32_t slots_collide_update_ = 4;
	uint32_t slots_calculate_normals_ = 4;
	uint32_t slots_per_compute_ =
		1 
		+ datas_.substeps * 
		(slots_integrate_clear + slots_spatial_hashing_
			+ datas_.iterations * slots_per_iteration_ 
			+ slots_collide_update_)
		+ slots_calculate_normals_
		+ 1;
	float pass_total_time_ = 0.0f;

	std::array<std::string, 19> labels_ = { "Intergrate", "ClearLambdas", "HashBuild", "RadixSort", "BuildCell", "BuildNeighbor", "SolveStretch", "SolveShear", "SolveBend", "SolveArea", "SolveSoftbodyStretch", "SolveSoftbodyVolume","SolveSelfCollision", "SolveInterCollision", "ApplyDeltas", "CollideSdf", "Update", "CalculateNormals", "Total" };
	std::unordered_map<std::string, double> label_time_;
	std::unordered_map<std::string, double> label_avg_time_;
	uint32_t time_count_ = 0;

	vku::Count counts_;
	vk::raii::DescriptorPool descriptor_pool_{ nullptr };

	struct RadixSortContext
	{
		VrdxSorter sorter = VK_NULL_HANDLE;
		vk::raii::Buffer       storage_buffer{ nullptr };
		vk::raii::DeviceMemory storage_memory{ nullptr };
		vk::DeviceSize         storage_size = 0;
	} radix_;

	struct PushConstant {
		struct Solve {
			uint32_t base;
			uint32_t count;
			float compliance;
			float beta;
		} solve;
		static_assert(sizeof(Solve) % 4 == 0, "push constant must be multiple of 4 bytes");

		struct MouseInteract {
			glm::vec3 ray_origin;
			uint32_t select_mode; // 0: none, 1: select, 2: drag
			glm::vec3 ray_dir;
			float radius = 0.1f;
			uint32_t depth_mode;
			uint32_t p0;
			uint32_t p1;
			uint32_t p2;
		} mouse_interact;
		static_assert(sizeof(MouseInteract) % 4 == 0, "push constant must be multiple of 4 bytes");
	} push_constants_;

	struct SetLayout {
		vk::raii::DescriptorSetLayout sim_params{ nullptr };
		vk::raii::DescriptorSetLayout cloth_compute{ nullptr };
	} set_layouts_;

	struct Set {
		vk::raii::DescriptorSet sim_params{ nullptr };
		vk::raii::DescriptorSet cloth_compute{ nullptr };
	} sets_;

	struct PipelineLayout {
		vk::raii::PipelineLayout common{ nullptr };
	} pipeline_layouts_;

	struct Pipeline {
		vk::raii::Pipeline wind{ nullptr };
		vk::raii::Pipeline integrate{ nullptr };
		vk::raii::Pipeline clear_lambdas{ nullptr };
		vk::raii::Pipeline solve_stretch{ nullptr };
		vk::raii::Pipeline solve_softbody_stretch{ nullptr };
		vk::raii::Pipeline solve_softbody_volume{ nullptr };
		vk::raii::Pipeline solve_shear{ nullptr };
		vk::raii::Pipeline solve_bend{ nullptr };
		vk::raii::Pipeline solve_area{ nullptr };
		vk::raii::Pipeline solve_self_collision{ nullptr };
		vk::raii::Pipeline solve_inter_cloth_collision{ nullptr };
		vk::raii::Pipeline apply_deltas{ nullptr };
		vk::raii::Pipeline collide_sdf{ nullptr };
		vk::raii::Pipeline update_velocity{ nullptr };

		vk::raii::Pipeline build_hash{ nullptr };
		vk::raii::Pipeline build_cell{ nullptr };
		vk::raii::Pipeline build_neighbor{ nullptr };

		vk::raii::Pipeline tri_normal{ nullptr };
		vk::raii::Pipeline vector_normal{ nullptr };

	} pipelines_;

private:
	void CreateCommandBuffers();
	void CreateQueryPool();
	void CreateDescriptorSetLayout();
	void CreateDescriptorPools();
	void CreateUniformBuffers();
	void CreateClothConstraintDatas();
	void CreateSoftBodyConstraintDatas();
	void CreateColiiders();
	void CreateSSBOBuffers();
	void CreateDescriptorSets();
	void CreateComputePipelineLayouts();
	void CreateComputePipelines();
	void CreateVrdxSorter();
};