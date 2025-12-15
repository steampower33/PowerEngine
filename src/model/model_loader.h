#pragma once

class TextureManager;
struct Vertex;

#include "model_data.h"
#include "vulkan_utils.h"

class ModelLoader {
public:
	ModelLoader();
	ModelLoader(const ModelLoader& rhs) = delete;
	ModelLoader(ModelLoader&& rhs) = delete;
	ModelLoader& operator=(const ModelLoader& rhs) = delete;
	ModelLoader& operator=(ModelLoader&& rhs) = delete;
	~ModelLoader();

	template<typename T>
	std::vector<T> ReadAccessor(const tinygltf::Model& model, int accessorIndex);

	void LoadModel(std::string modelPath, vku::VertexIncludeInfo vertexIncludeInfo, TextureManager& textureManager, float scale);
	void ParseMesh(const tinygltf::Model& model, vku::VertexIncludeInfo& vertexIncludeInfo, float scale);
	void ParseNodes(const tinygltf::Model& model);
	void ParseSkins(const tinygltf::Model& model);
	void ParseAnimations(const tinygltf::Model& model);

	Mesh mesh_;
	std::vector<Node> node_;
	std::vector<Skin> skin_;
	std::vector<AnimationClip> animations_;
};