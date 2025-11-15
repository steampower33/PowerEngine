#pragma once

class Texture2D;

class TextureManager
{
public:
	TextureManager(Context& context, GraphicsContext& graphicsContext);
	TextureManager(const TextureManager& rhs) = delete;
	TextureManager(TextureManager&& rhs) = delete;
	TextureManager& operator=(const TextureManager& rhs) = delete;
	TextureManager& operator=(TextureManager&& rhs) = delete;
	~TextureManager();

	std::unique_ptr<Texture2D> texture_{ nullptr };
};