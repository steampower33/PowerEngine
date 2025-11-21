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

	uint32_t timestamp_count_ = 10;

	const uint32_t nx_ = 64;
	const uint32_t ny_ = 64;

	glm::vec2 cloth_size_{ 2.0f, 2.0f };
	float spacing_x_ = cloth_size_.x / nx_;
	float spacing_y_ = cloth_size_.y / ny_;
	float cloth_height_ = 4.0f;
	float mass_ = cloth_size_.x * cloth_size_.y * 0.1f;

	struct Compliance {
		float stretch = 1e-6f;
		float diagonal = 1e-6f;
		float bend = 1e-3f;
	} compliance_;

	uint32_t particles_size_ = nx_ * ny_;
	uint32_t indices_size_ = 0;
	uint32_t edge_size = 0;
	uint32_t bend_size = 0;

	float deviding_dt_ = 120.0f;
	int iterations_ = 10;

	vku::Counts counts_;
	vk::raii::DescriptorPool descriptor_pool_{ nullptr };

	vk::raii::Buffer index_buffer_{ nullptr };
	vk::raii::DeviceMemory index_buffer_memory_{ nullptr };

	struct Data {
		std::vector<glm::vec4> positions;
		std::vector<glm::vec4> pred_positions;
		std::vector<glm::vec4> velocities;
		std::vector<float> inverse_masses;

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

		struct Bend {
			uint32_t p1, p2, p3, p4;
			float restAngle;
			float lambda;
			glm::vec2 pad;
		};
		static_assert(sizeof(Bend) == 32, "Bend must be 32 bytes");
		std::vector<Data::Bend> bends;

		struct SDFCollider {
			int   type;
			glm::vec3  center;
			float radius;
			glm::vec3  normal;
			glm::vec3  velocity;
			float pad;
		};
		static_assert(sizeof(SDFCollider) == 48, "SDFCollider must be 48 bytes");
		std::vector<SDFCollider> sdfColliders;
	} datas_;

	// |===== Push Constant =====|
	struct ClothPC {
		uint32_t nx1;
		uint32_t ny1;
	} cloth_pc_;
	static_assert(sizeof(ClothPC) % 4 == 0, "push constant must be multiple of 4 bytes");

	struct UBOData {
		struct SimParams {
			alignas(4)  float dt = 0.0f;
			alignas(4)  int   numParticles;
			alignas(4)  int   numEdges;
			alignas(4)  int   windTest = 0;
			alignas(4)  float windStrength = 1.0f;
			alignas(4)  float sphereRadius;
			alignas(4)  float maxSpeed;
			alignas(4)  float damping = 1.0f;
			alignas(4)  float relaxationFactor = 0.001f;
			alignas(4)  int   numBends;
			alignas(4)  uint32_t numColliders = 1;
			alignas(4)  float collisionMargin = 0.1f;
			alignas(16) glm::vec4 sphereCenter;
			alignas(16) glm::vec4 windDir = glm::vec4(0.0f, 0.0f, -1.0f, 0.0f);
			alignas(16) glm::vec4 gravity = glm::vec4(0.0f, -9.8f, 0.0f, 0.0f);
			alignas(4) float thickness = 0.008f;
			alignas(4) float friction = 0.1f;
			alignas(4) float pad1;
			alignas(4) float pad2;
		} sim_params;
		static_assert(sizeof(UBOData::SimParams) % 16 == 0, "std140 must be 16-byte aligned.");

		struct Render {
			glm::vec4 albedo_use{ 1.0f, 1.0f, 1.0f, 0.0f };

			int albedoIdx = -1;
			int metallicIdx = -1;
			int normalIdx = -1;
			int roughnessIdx = -1;

			int aoIdx = -1;
			int heightIdx = -1;
			float metallicFactor = 0.0f;
			float roughnessFactor = 1.0f;

			float aoFactor = 1.0f;
			float heightFactor = 0.0f;
			uint32_t p0 = 0;
			uint32_t p1 = 0;

			uint32_t albedoEnable = 0;
			uint32_t metallicEnable = 0;
			uint32_t normalEnable = 0;
			uint32_t roughtnessEnable = 0;

			uint32_t aoEnable = 0;
			uint32_t heightEnable = 0;
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
		vk::raii::Pipeline solve_atomic{ nullptr };
		vk::raii::Pipeline solve_bend{ nullptr };
		vk::raii::Pipeline apply_deltas{ nullptr };
		vk::raii::Pipeline collide_sdf{ nullptr };
		vk::raii::Pipeline update_velocity{ nullptr };
		vk::raii::Pipeline solve_shear{ nullptr };

		vk::raii::Pipeline cloth_solid{ nullptr };
		vk::raii::Pipeline cloth_wireframe{ nullptr };
		vk::raii::Pipeline cloth_point{ nullptr };

	} pipelines_;

	struct SSBO {
		vk::raii::Buffer position{ nullptr };
		vk::raii::Buffer velocity{ nullptr };
		vk::raii::Buffer inverse_mass{ nullptr };
		vk::raii::Buffer delta_x{ nullptr };
		vk::raii::Buffer delta_y{ nullptr };
		vk::raii::Buffer delta_z{ nullptr };
		vk::raii::Buffer dcount{ nullptr };
		vk::raii::Buffer edge{ nullptr };
		vk::raii::Buffer pred_position{ nullptr };
		vk::raii::Buffer bend{ nullptr };
		vk::raii::Buffer shear{ nullptr };
	} ssbos_;

	struct SSBOMemory {
		vk::raii::DeviceMemory position{ nullptr };
		vk::raii::DeviceMemory velocity{ nullptr };
		vk::raii::DeviceMemory inverse_mass{ nullptr };
		vk::raii::DeviceMemory delta_x{ nullptr };
		vk::raii::DeviceMemory delta_y{ nullptr };
		vk::raii::DeviceMemory delta_z{ nullptr };
		vk::raii::DeviceMemory dcount{ nullptr };
		vk::raii::DeviceMemory edge{ nullptr };
		vk::raii::DeviceMemory pred_position{ nullptr };
		vk::raii::DeviceMemory bend{ nullptr };
		vk::raii::DeviceMemory shear{ nullptr };

	} ssbo_memories_;

	struct SSBOSize {
		uint32_t position = 0;
		uint32_t velocity = 0;
		uint32_t inverse_mass = 0;
		uint32_t delta_x = 0;
		uint32_t delta_y = 0;
		uint32_t delta_z = 0;
		uint32_t dcount = 0;
		uint32_t edge = 0;
		uint32_t pred_position = 0;
		uint32_t bend = 0;
		uint32_t shear = 0;
	} ssbo_size_;

	struct Staging {
		vk::raii::Buffer position{ nullptr };
		vk::raii::Buffer velocity{ nullptr };
		vk::raii::Buffer inverse_mass{ nullptr };
		vk::raii::Buffer edge{ nullptr };
		vk::raii::Buffer pred_position{ nullptr };
		vk::raii::Buffer bend{ nullptr };
	} staging_;

	struct StagingMemory {
		vk::raii::DeviceMemory position{ nullptr };
		vk::raii::DeviceMemory velocity{ nullptr };
		vk::raii::DeviceMemory inverse_mass{ nullptr };
		vk::raii::DeviceMemory edge{ nullptr };
		vk::raii::DeviceMemory pred_position{ nullptr };
		vk::raii::DeviceMemory bend{ nullptr };

	} staging_memories_;

	struct StagingMapped {
		void* position{ nullptr };
		void* velocity{ nullptr };
		void* inverse_mass{ nullptr };
		void* edge{ nullptr };
		void* pred_position{ nullptr };
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

};