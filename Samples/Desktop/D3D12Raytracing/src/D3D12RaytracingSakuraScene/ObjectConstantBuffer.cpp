#include "stdafx.h"
#include "RaytracingHlslCompat.h"
#include "ObjectConstantBuffer.h"
#include "DirectXRaytracingHelper.h"
#include "D3D12RaytracingSakuraScene.h"
#include "ObjLoader.h"


void ObjectConstantBuffer::LoadObjMesh(
	std::string name,
	float scale,
	ObjLoader* loader,
	DX::DeviceResources* deviceResources,
	UINT descriptorSize,
	XMMATRIX transform,
	std::vector<Vertex>* vertices,
	std::vector<Index>* indices)
{
	loader->GetObjectVerticesAndIndices(name, scale, vertices, indices);
}