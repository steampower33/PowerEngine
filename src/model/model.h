#pragma once

class Context;
class PassManager;
struct Vertex;
struct Camera;
class TextureManager;

#include "vulkan_utils.h"
#include "mesh_data.h"

class Model
{
public:
	Model(std::string& modelPath, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, TextureManager& textureManager, glm::vec3 initPos, glm::quat initRotation, glm::vec4 colorUse, bool moveble, std::string name, float scale);
	Model(MeshData& meshData, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, glm::vec3 initPos, glm::quat initRotation, glm::vec4 colorUse, bool moveble, std::string name);
	Model(const Model& rhs) = delete;
	Model(Model&& rhs) = delete;
	Model& operator=(const Model& rhs) = delete;
	Model& operator=(Model&& rhs) = delete;
	~Model() = default;

	void LoadModel(const std::string& modelPath, const vku::VertexIncludeInfo& vertexIncludeInfo, TextureManager& textureManager, float scale);

	glm::mat4 world_{ 1.0f };
	glm::vec3 position_{ 0.0f, 0.0f, 0.0f };
	glm::quat rotation_{ 1.0f, 0.0f, 0.0f, 0.0f };
	glm::vec3 scale_{ 1.0f, 1.0f, 1.0f };
	float radius_ = 1.0f;
	bool checker_board_enable_ = false;
	bool movable_ = false;

	MeshData mesh_data_;

	std::string name_;
	glm::vec4 albedo_{ 0.0f };

	struct TextureIdx {
		int albedo = -1;
		int metallic = -1;
		int normal = -1;
		int roughness = -1;
		int ao = -1;
		int height = -1;
		int emissive = -1;
	} texture_idx_;

	struct TextureEnable {
		bool albedo = 0;
		bool metallic = 0;
		bool normal = 0;
		bool roughness = 0;
		bool ao = 0;
		bool height = 0;
	} texture_enable_;

	struct Factor {
		float metallic = 0.0f;
		float roughness = 1.0f;
		float ao = 1.0f;
		float height = 0.0f;
		float sheen_weight = 0.7f;
		float sheen_roughness = 1.0f;
	} factors_;

	void ApplyTransform(glm::vec3 scaleDelta, glm::quat rotationDelta, glm::vec3 translationDelta);

};