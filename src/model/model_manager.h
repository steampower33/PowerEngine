#pragma once

class Model;
class Context;
class PassManager;
class GeometryGenerator;
class TextureManager;
class Skybox;

class ModelManager
{
public:
	ModelManager(Context& context, TextureManager& textureManager);
	ModelManager(const ModelManager& rhs) = delete;
	ModelManager(ModelManager&& rhs) = delete;
	ModelManager& operator=(const ModelManager& rhs) = delete;
	ModelManager& operator=(ModelManager&& rhs) = delete;
	~ModelManager();

	static constexpr uint32_t kMaxObjects = 4;
	std::vector<std::unique_ptr<Model>> models_;
	//std::unique_ptr<Skybox> skybox_;
	std::unique_ptr<Model> soft_body_;
};