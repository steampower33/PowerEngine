#pragma once

class Model;
class Context;
class GraphicsContext;
class GeometryGenerator;
class TextureManager;
class Skybox;

class ModelManager
{
public:
	ModelManager(Context& context, GraphicsContext& graphicsContext, TextureManager& textureManager);
	ModelManager(const ModelManager& rhs) = delete;
	ModelManager(ModelManager&& rhs) = delete;
	ModelManager& operator=(const ModelManager& rhs) = delete;
	ModelManager& operator=(ModelManager&& rhs) = delete;
	~ModelManager();

	static constexpr uint32_t kMaxObjects = 2;
	std::vector<std::unique_ptr<Model>> models_;
	//std::unique_ptr<Skybox> skybox_;
};