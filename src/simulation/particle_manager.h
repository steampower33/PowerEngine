#pragma once

class Context;
class Model;
class ModelManager;
class Texture;
class TextureManager;
class ModelLoader;

#include "model_data.h"

struct Cloth
{
	std::string name;

	float spacing;
	float gsm;
	glm::vec2 cloth_size;
	uint32_t nx;
	uint32_t ny;
	uint32_t nx1;
	uint32_t ny1;
	glm::vec4 color;

	glm::vec3 origin;
	float     angle_deg;
	glm::vec3 axis;

	uint32_t offset_particle;
	uint32_t offset_indices;

	uint32_t num_particle;
	uint32_t num_indices;

	ubo_data::Model ubo_data;

	enum Type {
		PLANE,
		MESH
	} type;

	bool render;

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
};

struct SoftBody
{
	std::string name;

	vku::TetMesh tetmesh;

	glm::vec4 color;

	uint32_t offset_particle = 0;
	uint32_t offset_indices = 0;

	uint32_t num_particle = 0;
	uint32_t num_indices = 0;

	glm::vec3 origin;
	float     angle_deg;
	glm::vec3 axis;

	bool render;

	ubo_data::Model ubo_data;
};

class ParticleManager
{
public:
	ParticleManager(Context& context, ModelManager& modelManager, TextureManager& textureManager);
	ParticleManager(const ParticleManager& rhs) = delete;
	ParticleManager(ParticleManager&& rhs) = delete;
	ParticleManager& operator=(const ParticleManager& rhs) = delete;
	ParticleManager& operator=(ParticleManager&& rhs) = delete;
	~ParticleManager();

	void SetPlaneCloth(Cloth& cloth);
	void SetClothFromMesh(Cloth& cloth);
	void SetSoftbody(std::string path, SoftBody& softbody, bool isJson);
	void ResetCloth(Cloth& cloth);
	void ResetSoftbody(SoftBody& softbody);
	void ResetVolumeConstraint();

	Context& context_;

	std::vector<uint32_t> sim_base_;
	std::vector<uint32_t> sim_count_;

	uint32_t total_particles_ = 0;
	uint32_t total_indices_ = 0;
	uint32_t total_tries_ = 0;

	uint32_t num_cloth_particles_ = 0;
	uint32_t num_cloth_indices_ = 0;
	std::vector<Cloth> clothes_;

	uint32_t num_softbody_particles_ = 0;
	uint32_t num_softbody_indices_ = 0;
	//SoftBody soft_body_;
	std::vector<SoftBody> softbodies_;

	float default_cloth_spacing_ = 0.02f;

	std::vector<glm::vec4> positions_;
	std::vector<glm::vec4> pred_positions_;
	std::vector<glm::vec4> velocities_;
	std::vector<float> inverse_masses_;
	std::vector<float> masses_;
	std::vector<uint32_t> indices_;
	std::vector<glm::vec4> normals_;

	struct Volume {
		uint32_t i0;
		uint32_t i1;
		uint32_t i2;
		uint32_t i3;
		float rest_volume;
		float lambda;
		float p1;
		float p2;
	};
	std::vector<Volume> volume_constraints;

	uint32_t object_cnt_ = 0;

	enum ObjectType {
		CLOTH,
		SOFTBODY
	};

	struct ColiisiotnMask {
		uint32_t object_id;
		uint32_t object_type;
	};

	std::vector<ColiisiotnMask> collision_masks_;

	std::vector<glm::vec4> tri_normals_;
	std::vector<uint32_t> vertex_tri_offsets_;  // size = numVertices + 1
	std::vector<uint32_t> vertex_tri_indices_;  // size = numTris * 3

	vk::raii::Buffer index_buffer_{ nullptr };
	vk::raii::DeviceMemory index_buffer_memory_{ nullptr };

	std::vector<uint32_t> particle_hashes_;
	std::vector<uint32_t> particle_indices_;
	std::vector<uint32_t> starts_;
	std::vector<uint32_t> ends_;

	uint32_t num_neighbors_;
	std::vector<uint32_t> neighbors_;
	std::vector<uint32_t> neighbor_lambdas_;

	struct SSBO {
		vk::raii::Buffer position{ nullptr };
		vk::raii::Buffer pred_position{ nullptr };
		vk::raii::Buffer velocity{ nullptr };
		vk::raii::Buffer inverse_mass{ nullptr };

		vk::raii::Buffer particle_hash{ nullptr };
		vk::raii::Buffer sorted_indice{ nullptr };
		vk::raii::Buffer start{ nullptr };
		vk::raii::Buffer end{ nullptr };
		vk::raii::Buffer neighbor{ nullptr };
		vk::raii::Buffer neighbor_lambda{ nullptr };

		vk::raii::Buffer index{ nullptr };
		vk::raii::Buffer normal{ nullptr };

		vk::raii::Buffer tri_normals{ nullptr };
		vk::raii::Buffer vertex_tri_offsets{ nullptr };
		vk::raii::Buffer vertex_tri_indices{ nullptr };

		vk::raii::Buffer collision_masks_{ nullptr };
	} ssbos_;

	struct SSBOMemory {
		vk::raii::DeviceMemory position{ nullptr };
		vk::raii::DeviceMemory pred_position{ nullptr };
		vk::raii::DeviceMemory velocity{ nullptr };
		vk::raii::DeviceMemory inverse_mass{ nullptr };

		vk::raii::DeviceMemory particle_hash{ nullptr };
		vk::raii::DeviceMemory sorted_indice{ nullptr };
		vk::raii::DeviceMemory start{ nullptr };
		vk::raii::DeviceMemory end{ nullptr };
		vk::raii::DeviceMemory neighbor{ nullptr };
		vk::raii::DeviceMemory neighbor_lambda{ nullptr };

		vk::raii::DeviceMemory index{ nullptr };
		vk::raii::DeviceMemory normal{ nullptr };

		vk::raii::DeviceMemory tri_normals{ nullptr };
		vk::raii::DeviceMemory vertex_tri_offsets{ nullptr };
		vk::raii::DeviceMemory vertex_tri_indices{ nullptr };

		vk::raii::DeviceMemory collision_masks_{ nullptr };
	} ssbo_memories_;

	struct SSBOSize {
		vk::DeviceSize position = 0;
		vk::DeviceSize pred_position = 0;
		vk::DeviceSize velocity = 0;
		vk::DeviceSize inverse_mass = 0;

		vk::DeviceSize particle_hash = 0;
		vk::DeviceSize sorted_indice = 0;
		vk::DeviceSize start = 0;
		vk::DeviceSize end = 0;
		vk::DeviceSize neighbor = 0;
		vk::DeviceSize neighbor_lambda = 0;

		vk::DeviceSize index = 0;
		vk::DeviceSize normal = 0;
		vk::DeviceSize tri_normals = 0;
		vk::DeviceSize vertex_tri_offsets = 0;
		vk::DeviceSize vertex_tri_indices = 0;

		vk::DeviceSize collision_masks_ = 0;
	} ssbo_size_;

	struct Staging {
		vk::raii::Buffer position{ nullptr };
		vk::raii::Buffer pred_position{ nullptr };
		vk::raii::Buffer velocity{ nullptr };
		vk::raii::Buffer inverse_mass{ nullptr };
	} staging_;

	struct StagingMemory {
		vk::raii::DeviceMemory position{ nullptr };
		vk::raii::DeviceMemory pred_position{ nullptr };
		vk::raii::DeviceMemory velocity{ nullptr };
		vk::raii::DeviceMemory inverse_mass{ nullptr };
	} staging_memories_;

	struct StagingMapped {
		void* position{ nullptr };
		void* pred_position{ nullptr };
		void* velocity{ nullptr };
		void* inverse_mass{ nullptr };
	} staging_mapped_;

private:
	void CreateSSBO();
	void BuildVertexAdjacency();
};