#pragma once

class Context;
class GraphicsContext;
struct Vertex;
struct Camera;
class TextureManager;

#include "vulkan_utils.h"
#include "mesh_data.h"

class Model
{
public:
	Model(const std::string& modelPath, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, GraphicsContext& graphicsContext, TextureManager& textureManager, uint32_t& model_count, glm::vec3 initPos, glm::quat initRotation, glm::vec4 colorUse, bool moveble);
	Model(MeshData& meshData, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, GraphicsContext& graphicsContext, uint32_t& model_count, glm::vec3 initPos, glm::quat initRotation, glm::vec4 colorUse, bool moveble);
	Model(const Model& rhs) = delete;
	Model(Model&& rhs) = delete;
	Model& operator=(const Model& rhs) = delete;
	Model& operator=(Model&& rhs) = delete;
	~Model() = default;

	glm::vec4 albedo_use_{ 0.0f };

	void LoadModel(const std::string& modelPath, const vku::VertexIncludeInfo& vertexIncludeInfo, TextureManager& textureManager);

	glm::mat4 world_{ 1.0f };
	glm::vec3 position_{ 0.0f, 0.0f, 0.0f };
	glm::quat rotation_{ 1.0f, 0.0f, 0.0f, 0.0f };
	glm::vec3 scale_{ 1.0f, 1.0f, 1.0f };
	float radius_ = 1.0f;

	bool moveble_ = false;

	MeshData mesh_data_;

	struct TextureIdx {
		uint32_t albedo = 0;
		uint32_t metallic = 0;
		uint32_t normal = 0;
		uint32_t roughness = 0;
		uint32_t ao = 0;
		uint32_t height = 0;
		uint32_t emissive = 0;
	} texture_idx_;

	struct TextureUse {
		uint32_t albedo = 0;
		uint32_t metallic = 0;
		uint32_t normal = 0;
		uint32_t roughtness = 0;
		uint32_t ao = 0;
		uint32_t height = 0;
	} texture_use_;

	struct Factor {
		float metallic = 0.0f;
		float roughness = 0.7f;
		float ao = 1.0f;
	} factors_;

	void ApplyTransform(const glm::quat& rotationDelta, const glm::vec3& translationDelta);

};