#include "swapchain.hpp"
#include "context.hpp"

Swapchain::Swapchain(Context* ctx)
    : ctx(ctx)
{

}

Swapchain::~Swapchain()
{
}

bool Swapchain::draw()
{
    return false;
}
