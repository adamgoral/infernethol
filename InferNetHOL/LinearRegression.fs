namespace InferNetHOL
module LinearRegression =
    open MicrosoftResearch.Infer
    open MicrosoftResearch.Infer.Fun.FSharp.Inference
    open MicrosoftResearch.Infer.Fun.FSharp.Syntax
    open MicrosoftResearch.Infer.Distributions

    [<ReflectedDefinition>]
    let pointNearLine x a b invNoise =
        let y = random(GaussianFromMeanAndPrecision(a * x + b, invNoise))
        x, y

    [<ReflectedDefinition>]
    let parametersPrior () =
        let a = random(GaussianFromMeanAndPrecision(0.,0.01))
        let b = random(GaussianFromMeanAndPrecision(0.,0.01))
        let invNoise = random(GammaFromShapeAndScale(1.,5.))
        a, b, invNoise

    [<ReflectedDefinition>]
    let model data =
        let a, b, invNoise = parametersPrior ()
        observe(data = [|for (x, _) in data -> (x, random(GaussianFromMeanAndPrecision(a * x + b, invNoise)))|])
        a, b, invNoise

    let inferFromData (data : (float * float) []) =
        let (aD: Gaussian), 
            (bD: Gaussian), 
            (noiseD: Gamma) = infer <@ model @> data
        aD, bD, noiseD