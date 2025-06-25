.\dxc.exe -Zi -T lib_6_9 -Vn g_pPathtracer -Fh obj\x64\Debug\CompiledShaders\Pathtracer.hlsl.h .\SampleCore\Shaders\Pathtracer.hlsl
.\dxc.exe -T lib_6_9 -Vn g_pRTAO -Fh obj\x64\Debug\CompiledShaders\RTAO.hlsl.h .\RTAO\Shaders\RTAO.hlsl
.\dxc.exe -T cs_6_6 -Vn g_pAORayGenCS -Fh obj\x64\Debug\CompiledShaders\AORayGenCS.hlsl.h .\RTAO\Shaders\AORayGenCS.hlsl
.\dxc.exe -T cs_6_6 -Vn g_pCalculateMeanVarianceCS -Fh obj\x64\Debug\CompiledShaders\CalculateMeanVarianceCS.hlsl.h .\RTAO\Shaders\Denoising\CalculateMeanVarianceCS.hlsl
.\dxc.exe -T cs_6_6 -Vn g_pCalculatePartialDerivativesViaCentralDifferencesCS -Fh obj\x64\Debug\CompiledShaders\CalculatePartialDerivativesViaCentralDifferencesCS.hlsl.h .\SampleCore\Shaders\util\CalculatePartialDerivativesViaCentralDifferencesCS.hlsl
.\dxc.exe -T cs_6_6 -Vn g_pCompositionCS -Fh obj\x64\Debug\CompiledShaders\CompositionCS.hlsl.h .\SampleCore\Shaders\CompositionCS.hlsl
.\dxc.exe -T cs_6_6 -Vn g_pCountingSort_SortRays_64x128rayGroupCS -Fh obj\x64\Debug\CompiledShaders\CountingSort_SortRays_64x128rayGroupCS.hlsl.h ".\RTAO\Shaders\Ray sorting\CountingSort_SortRays_64x128rayGroupCS.hlsl"
.\dxc.exe -T cs_6_6 -Vn g_pDisocclusionBlur3x3CS -Fh obj\x64\Debug\CompiledShaders\DisocclusionBlur3x3CS.hlsl.h .\RTAO\Shaders\Denoising\DisocclusionBlur3x3CS.hlsl
.\dxc.exe -T cs_6_6 -Vn g_pDownsampleGBufferDataBilateralFilter2x2CS -Fh obj\x64\Debug\CompiledShaders\DownsampleGBufferDataBilateralFilter2x2CS.hlsl.h .\SampleCore\Shaders\util\DownsampleGBufferDataBilateralFilter2x2CS.hlsl
.\dxc.exe -T cs_6_6 -Vn g_pEdgeStoppingFilter_Gaussian3x3CS -Fh obj\x64\Debug\CompiledShaders\EdgeStoppingFilter_Gaussian3x3CS.hlsl.h .\RTAO\Shaders\Denoising\EdgeStoppingFilter_Gaussian3x3CS.hlsl
.\dxc.exe -T cs_6_6 -Vn g_pEdgeStoppingFilter_Gaussian5x5CS -Fh obj\x64\Debug\CompiledShaders\EdgeStoppingFilter_Gaussian5x5CS.hlsl.h .\RTAO\Shaders\Denoising\EdgeStoppingFilter_Gaussian5x5CS.hlsl
.\dxc.exe -T cs_6_6 -Vn g_pFillInCheckerboard_CrossBox4TapFilterCS -Fh obj\x64\Debug\CompiledShaders\FillInCheckerboard_CrossBox4TapFilterCS.hlsl.h .\RTAO\Shaders\Denoising\FillInCheckerboard_CrossBox4TapFilterCS.hlsl
.\dxc.exe -T cs_6_6 -Vn g_pGaussianFilter3x3CS -Fh obj\x64\Debug\CompiledShaders\GaussianFilter3x3CS.hlsl.h .\RTAO\Shaders\Denoising\GaussianFilter3x3CS.hlsl
.\dxc.exe -T cs_6_6 -Vn g_pGaussianFilterRG3x3CS -Fh obj\x64\Debug\CompiledShaders\GaussianFilterRG3x3CS.hlsl.h .\RTAO\Shaders\Denoising\GaussianFilterRG3x3CS.hlsl
.\dxc.exe -T cs_6_6 -Vn g_pGenerateGrassStrawsCS -Fh obj\x64\Debug\CompiledShaders\GenerateGrassStrawsCS.hlsl.h .\SampleCore\Shaders\util\GenerateGrassStrawsCS.hlsl
.\dxc.exe -T cs_6_6 -Vn g_pReduceSumFloatCS -Fh obj\x64\Debug\CompiledShaders\ReduceSumFloatCS.hlsl.h .\SampleCore\Shaders\util\ReduceSumFloatCS.hlsl
.\dxc.exe -T cs_6_6 -Vn g_pReduceSumUintCS -Fh obj\x64\Debug\CompiledShaders\ReduceSumUintCS.hlsl.h .\SampleCore\Shaders\util\ReduceSumUintCS.hlsl
.\dxc.exe -T cs_6_6 -Vn g_pTemporalSupersampling_BlendWithCurrentFrameCS -Fh obj\x64\Debug\CompiledShaders\TemporalSupersampling_BlendWithCurrentFrameCS.hlsl.h .\RTAO\Shaders\Denoising\TemporalSupersampling_BlendWithCurrentFrameCS.hlsl
.\dxc.exe -T cs_6_6 -Vn g_pTemporalSupersampling_ReverseReprojectCS -Fh obj\x64\Debug\CompiledShaders\TemporalSupersampling_ReverseReprojectCS.hlsl.h .\RTAO\Shaders\Denoising\TemporalSupersampling_ReverseReprojectCS.hlsl
.\dxc.exe -T cs_6_6 -Vn g_pUpsampleBilateralFilter2x2Float2CS -Fh obj\x64\Debug\CompiledShaders\UpsampleBilateralFilter2x2Float2CS.hlsl.h .\SampleCore\Shaders\util\UpsampleBilateralFilter2x2Float2CS.hlsl
.\dxc.exe -T cs_6_6 -Vn g_pUpsampleBilateralFilter2x2FloatCS -Fh obj\x64\Debug\CompiledShaders\UpsampleBilateralFilter2x2FloatCS.hlsl.h .\SampleCore\Shaders\util\UpsampleBilateralFilter2x2FloatCS.hlsl
.\dxc.exe -T cs_6_6 -Vn g_pUpsampleBilateralFilter2x2UintCS -Fh obj\x64\Debug\CompiledShaders\UpsampleBilateralFilter2x2UintCS.hlsl.h .\SampleCore\Shaders\util\UpsampleBilateralFilter2x2UintCS.hlsl