#include <DirectXMath.h>
#include <vector>
#include <string>
#include "ObjLoader.h"
#include "DeviceResources.h"
#include "RaytracingHlslCompat.h"


using namespace DirectX;

using Index = uint16_t; // or whatever your project uses

struct CubeConstantBuffer
{
    XMFLOAT4 albedo;
    uint32_t materialID;

    void Initialize(uint32_t material);

    void LoadCube(
        ObjLoader* loader,
        float xScale,
        float yScale,
        float zScale,
        float xTranslate,
        float yTranslate,
        float zTranslate,
        float uvScale,
        DX::DeviceResources* deviceResources,
        UINT descriptorSize,
        std::vector<Vertex>* floorVertices,
        std::vector<Index>* indices);

    void LoadObjMesh(
        std::string name,
        float scale,
        ObjLoader* loader,
        DX::DeviceResources* deviceResources,
        UINT descriptorSize,
        XMMATRIX transform,
        std::vector<Vertex>* floorVertices,
        std::vector<Index>* indices);
};
