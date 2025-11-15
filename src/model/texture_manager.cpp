
#include "texture_2d.h"

#include "texture_manager.h"

TextureManager::TextureManager(Context& context, GraphicsContext& graphicsContext)
{
	texture_ = std::make_unique<Texture2D>("assets/textures/vulkan_cloth_rgba.ktx", context, graphicsContext);
}

TextureManager::~TextureManager()
{

}