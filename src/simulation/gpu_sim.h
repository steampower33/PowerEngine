#pragma once

class Context;
class Swapchain;
class Texture;
class TextureManager;
class Model;
class ModelManager;

#include "vulkan_utils.h"

class GpuSim
{
public:
	GpuSim(
		Context& context,
		Swapchain& swapchain,
		TextureManager& textureManager,
		ModelManager& modelManager,
		vk::raii::DescriptorSetLayout& globalSetLayout,
		std::vector<vk::Format>& formats,
		vk::raii::DescriptorSetLayout& tex2DSetLayout);
	GpuSim(const GpuSim& rhs) = delete;
	GpuSim(GpuSim&& rhs) = delete;
	GpuSim& operator=(const GpuSim& rhs) = delete;
	GpuSim& operator=(GpuSim&& rhs) = delete;
	~GpuSim() = default;

	void UpdateComputeUBO(uint32_t currentFrame, std::unique_ptr<Model>& model);
	void UpdateGraphicsUBO(uint32_t currentFrame);

	void ComputeRecord(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::QueryPool& timestampPool, uint32_t& timestampSteps, vku::TestScene& testScene);
	void GraphicsRecord(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::DescriptorSet& globalSet, uint32_t globalOffset, vku::PolygonMode mode,
		vk::raii::DescriptorSet& tex2DSet);

	void CopyDatas(const vk::raii::CommandBuffer& cmd);
	void UpdateTestScene(const vk::raii::CommandBuffer& cmd, vku::TestScene& testScene);

	uint32_t timestamp_count_ = 12;

	const uint32_t nx_ = 100;
	const uint32_t ny_ = 100;

	glm::vec2 cloth_size_{ 4.0f, 4.0f };
	float spacing_x_ = cloth_size_.x / nx_;
	float spacing_y_ = cloth_size_.y / ny_;
	float cloth_height_ = 6.0f;
	float mass_ = 0.1f;

	struct Compliance {
		float stretch = 1e-7f;
		float diagonal = 1e-7f;
		float shear = 1.0e-6f;
		float bend = 1.0f;
	} compliance_;

	uint32_t particles_size_ = nx_ * ny_;
	uint32_t indices_size_ = 0;
	uint32_t edge_size_ = 0;
	uint32_t shear_size_ = 0;
	uint32_t bend_size_ = 0;

	float deviding_dt_ = 240.0f;
	int iterations_ = 20;

	vku::Counts counts_;
	vk::raii::DescriptorPool descriptor_pool_{ nullptr };

	vk::raii::Buffer index_buffer_{ nullptr };
	vk::raii::DeviceMemory index_buffer_memory_{ nullptr };

	struct Data {
		std::vector<glm::vec4> positions;
		std::vector<glm::vec4> pred_positions;
		std::vector<glm::vec4> velocities;
		std::vector<float> inverse_masses;
		std::vector<float> masses;
		std::vector<uint32_t> indices;

		struct Edge {
			uint32_t i;
			uint32_t j;
			float    rest;
			float    lambda;
		};
		static_assert(sizeof(Edge) == 16, "Edge must be 16 bytes");

		std::vector<Edge> edges;
		std::array<uint32_t, 6> pass_offsets;
		std::vector<std::pair<uint32_t, uint32_t>> passes[6];

		struct Shear {
			uint32_t i0, i1, i2;
			float rest_dot;
			float lambda;
			float p0, p1, p2;
		};
		static_assert(sizeof(Shear) == 32, "Shear must be 32 bytes");

		struct Bend {
			uint32_t i0, i1, i2, i3;
			float rest_angle;
			float lambda;
			glm::vec2 pad;
		};
		static_assert(sizeof(Bend) == 32, "Bend must be 32 bytes");
		std::vector<Data::Bend> bends;
	} datas_;

	struct EdgeKey
	{
		uint32_t a, b; // always a < b

		bool operator==(const EdgeKey& o) const noexcept {
			return a == o.a && b == o.b;
		}
	};

	struct EdgeKeyHash
	{
		size_t operator()(const EdgeKey& k) const noexcept {
			return (size_t(k.a) << 32) ^ size_t(k.b);
		}
	};

	struct TriRef
	{
		uint32_t triIndex;
		uint32_t oppVertex;
	};

	struct ClothPC {
		uint32_t nx1;
		uint32_t ny1;
	} cloth_pc_;
	static_assert(sizeof(ClothPC) % 4 == 0, "push constant must be multiple of 4 bytes");

	struct UBOData {
		struct SimParams {
			alignas(4)  float dt = 0.0f;
			alignas(4)  uint32_t num_particles;
			alignas(4)  uint32_t num_edges;
			alignas(4)  int windTest = 0;
			alignas(4)  float wind_strength = 1.0f;
			alignas(4)  float sphere_radius;
			alignas(4)  float max_speed;
			alignas(4)  float damping = 0.5f;
			alignas(4)  float relaxation_factor = 0.2f;
			alignas(4)  uint32_t num_bends;
			alignas(4)  uint32_t num_shears;
			alignas(4)  float collision_margin = 0.1f;
			alignas(16) glm::vec4 sphere_center;
			alignas(16) glm::vec4 wind_dir = glm::vec4(0.0f, 0.0f, -1.0f, 0.0f);
			alignas(16) glm::vec4 gravity = glm::vec4(0.0f, -9.8f, 0.0f, 0.0f);
			alignas(4) float thickness = 0.008f;
			alignas(4) float friction = 0.001f;
			alignas(4) float pad1;
			alignas(4) float pad2;
		} sim_params;
		static_assert(sizeof(UBOData::SimParams) % 16 == 0, "std140 must be 16-byte aligned.");

		struct Render {
			glm::vec4 albedo_use{ 1.0f, 1.0f, 1.0f, 0.0f };

			int albedo_idx = -1;
			int metallic_idx = -1;
			int normal_idx = -1;
			int roughness_idx = -1;

			int ao_idx = -1;
			int height_idx = -1;
			float metallic_factor = 0.0f;
			float roughness_factor = 1.0f;

			float ao_factor = 1.0f;
			float height_factor = 0.0f;
			uint32_t p0 = 0;
			uint32_t p1 = 0;

			uint32_t albedo_enable = 0;
			uint32_t metallic_enable = 0;
			uint32_t normal_enable = 0;
			uint32_t roughtnessEnable = 0;

			uint32_t ao_enable = 0;
			uint32_t height_enable = 0;
			uint32_t p3;
			uint32_t p4;
		} render;
		static_assert(sizeof(UBOData::Render) % 16 == 0, "std140 must be 16-byte aligned.");

	} ubo_data_;

	struct PushConstant {
		uint32_t base;
		uint32_t count;
		float compliance;
	} pc_;

	struct UBO {
		vk::raii::Buffer sim_params{ nullptr };
		vk::raii::Buffer render{ nullptr };
	} ubos_;

	struct UBOMemory {
		vk::raii::DeviceMemory sim_params{ nullptr };
		vk::raii::DeviceMemory render{ nullptr };
	} ubo_memories_;

	struct UBOMapped {
		void* sim_params{ nullptr };
		void* render{ nullptr };
	} ubo_mapped_;

	struct UBOSize {
		vk::DeviceSize sim_params;
		vk::DeviceSize render;
	} ubo_size_;

	struct SetLayout {
		vk::raii::DescriptorSetLayout sim_params{ nullptr };
		vk::raii::DescriptorSetLayout render{ nullptr };
		vk::raii::DescriptorSetLayout cloth_compute{ nullptr };
		vk::raii::DescriptorSetLayout cloth_graphics{ nullptr };
	} set_layouts_;

	struct Set {
		vk::raii::DescriptorSet sim_params{ nullptr };
		vk::raii::DescriptorSet render{ nullptr };
		vk::raii::DescriptorSet cloth_compute{ nullptr };
		vk::raii::DescriptorSet cloth_graphics{ nullptr };
	} sets_;

	struct PipelineLayout {
		vk::raii::PipelineLayout common{ nullptr };
		vk::raii::PipelineLayout cloth_graphics{ nullptr };
	} pipeline_layouts_;

	struct Pipeline {
		vk::raii::Pipeline clear_lambdas{ nullptr };
		vk::raii::Pipeline integrate{ nullptr };
		vk::raii::Pipeline solve_coloring{ nullptr };
		vk::raii::Pipeline solve_diagonal{ nullptr };
		vk::raii::Pipeline solve_shear{ nullptr };
		vk::raii::Pipeline solve_bend{ nullptr };
		vk::raii::Pipeline apply_deltas{ nullptr };
		vk::raii::Pipeline collide_sdf{ nullptr };
		vk::raii::Pipeline update_velocity{ nullptr };

		vk::raii::Pipeline cloth_solid{ nullptr };
		vk::raii::Pipeline cloth_wireframe{ nullptr };
		vk::raii::Pipeline cloth_point{ nullptr };

	} pipelines_;

	struct SSBO {
		vk::raii::Buffer position{ nullptr };
		vk::raii::Buffer pred_position{ nullptr };
		vk::raii::Buffer velocity{ nullptr };
		vk::raii::Buffer inverse_mass{ nullptr };
		vk::raii::Buffer delta_x{ nullptr };
		vk::raii::Buffer delta_y{ nullptr };
		vk::raii::Buffer delta_z{ nullptr };
		vk::raii::Buffer delta_count{ nullptr };
		vk::raii::Buffer edge{ nullptr };
		vk::raii::Buffer shear{ nullptr };
		vk::raii::Buffer bend{ nullptr };
	} ssbos_;

	struct SSBOMemory {
		vk::raii::DeviceMemory position{ nullptr };
		vk::raii::DeviceMemory pred_position{ nullptr };
		vk::raii::DeviceMemory velocity{ nullptr };
		vk::raii::DeviceMemory inverse_mass{ nullptr };
		vk::raii::DeviceMemory delta_x{ nullptr };
		vk::raii::DeviceMemory delta_y{ nullptr };
		vk::raii::DeviceMemory delta_z{ nullptr };
		vk::raii::DeviceMemory delta_count{ nullptr };
		vk::raii::DeviceMemory edge{ nullptr };
		vk::raii::DeviceMemory shear{ nullptr };
		vk::raii::DeviceMemory bend{ nullptr };
	} ssbo_memories_;

	struct SSBOSize {
		uint32_t position = 0;
		uint32_t pred_position = 0;
		uint32_t velocity = 0;
		uint32_t inverse_mass = 0;
		uint32_t delta_x = 0;
		uint32_t delta_y = 0;
		uint32_t delta_z = 0;
		uint32_t delta_count = 0;
		uint32_t edge = 0;
		uint32_t shear = 0;
		uint32_t bend = 0;
	} ssbo_size_;

	struct Staging {
		vk::raii::Buffer position{ nullptr };
		vk::raii::Buffer pred_position{ nullptr };
		vk::raii::Buffer velocity{ nullptr };
		vk::raii::Buffer inverse_mass{ nullptr };
		vk::raii::Buffer edge{ nullptr };
		vk::raii::Buffer bend{ nullptr };
	} staging_;

	struct StagingMemory {
		vk::raii::DeviceMemory position{ nullptr };
		vk::raii::DeviceMemory pred_position{ nullptr };
		vk::raii::DeviceMemory velocity{ nullptr };
		vk::raii::DeviceMemory inverse_mass{ nullptr };
		vk::raii::DeviceMemory edge{ nullptr };
		vk::raii::DeviceMemory bend{ nullptr };
	} staging_memories_;

	struct StagingMapped {
		void* position{ nullptr };
		void* pred_position{ nullptr };
		void* velocity{ nullptr };
		void* inverse_mass{ nullptr };
		void* edge{ nullptr };
		void* bend{ nullptr };
	} staging_mapped_;

private:
	void CreateDescriptorSetLayout(Context& context);
	void CreateDescriptorPools(Context& context);
	void CreateUniformBuffers(Context& context,
		ModelManager& modelManager);
	void CreateSSBOBuffers(Context& context);
	void CreateDescriptorSets(Context& context, TextureManager& textureManager);
	void CreateComputePipelines(Context& context);
	void CreateGraphicsPipelines(Context& context, vk::raii::DescriptorSetLayout& globalSetLayout,
		std::vector<vk::Format>& formats,
		vk::raii::DescriptorSetLayout& tex2DSetLayout);
	float ComputeRestBendAngle(
		uint32_t p1, uint32_t p2,
		uint32_t p3, uint32_t p4,
		const std::vector<glm::vec4>& pos);
	void BuildBendConstraintsFromTriangles();

};