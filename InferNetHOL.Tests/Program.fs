open MicrosoftResearch.Infer
open MicrosoftResearch.Infer.Fun.FSharp.Inference
open MicrosoftResearch.Infer.Fun.FSharp.Syntax
open MicrosoftResearch.Infer.Fun.Lib
open MicrosoftResearch.Infer.Models
open MicrosoftResearch.Infer.Distributions
open MicrosoftResearch.Infer.Maths
open System
open Coins

let randomSequence distribution = Seq.initInfinite (fun _ -> random distribution)

[<ReflectedDefinition>]
let gaussianParameters data =
    let meanPrior = random (GaussianFromMeanAndPrecision (0., 0.01))
    let variancePrior = random (GammaFromShapeAndScale (1., 1.))
    observe (data = Array.init data.Length (fun _ -> random(GaussianFromMeanAndVariance (meanPrior, variancePrior))))
    meanPrior, variancePrior

let inferGaussianFromData data =
    let (mean: Gaussian), (precision: Gamma) = infer <@gaussianParameters@> data
    Gaussian (mean.GetMean(), precision.GetMean())

[<ReflectedDefinition>]
let gaussianVectorParameters (data: Vector[]) =
    let mean = random(VectorGaussianFromMeanAndPrecision(VectorFromArray [|for i in 0..data.Length -> 0.|], IdentityScaledBy(data.Length, 0.01)))
    let covariance = random(WishartFromShapeAndScale(100.0, IdentityScaledBy(data.Length, 0.01)))
    observe (data = Array.init data.Length (fun _ -> random(VectorGaussianFromMeanAndPrecision(mean, covariance))))
    mean, covariance

let inferVectorGaussianFromData (data: Vector[]) =
    let (mean: VectorGaussian), (covariance: Wishart) = infer <@gaussianVectorParameters@> data
    VectorGaussian(mean.GetMean(), covariance.GetMean())

[<ReflectedDefinition>]
let vectorWeightsCovarianceAndNoise (xs: Vector[]) (ys: float[]) =
    let noise = random(GammaFromShapeAndScale(1.,1.))
    let weightsMeans = random(VectorGaussianFromMeanAndPrecision(VectorFromArray [|for i in 0..xs.Length -> 0.|], IdentityScaledBy(xs.Length, 0.01)))
    let weightsCovariance = random(WishartFromShapeAndScale(100., IdentityScaledBy(xs.Length, 0.01)))
    let weights = random(VectorGaussianFromMeanAndPrecision(weightsMeans, weightsCovariance))
    observe (ys = [|for x in xs -> random(GaussianFromMeanAndPrecision(InnerProduct (x, weights), noise))|])
    weights, weightsCovariance, noise

let simpleGaussianInference () =
    let observed = [13.; 15.; 17.; 5.; 20.; 13.; 16.5; 22.;20.;19.;13.;13.;14.]
    let noise =  Variable.GammaFromShapeAndScale(2., 5.)
    let avg = List.average observed
    let avgVar = Variable.GaussianFromMeanAndPrecision(avg, 0.01)
    let observedVars = List.map (fun o -> let oVar = Variable.GaussianFromMeanAndPrecision (avgVar, noise)
                                          oVar.ObservedValue <- o) observed
    let engine = InferenceEngine.DefaultEngine
    let averagePosterior = engine.Infer<Gaussian> avgVar
    let noisePosterior = engine.Infer<Gamma> noise
    let estVar = Variable.GaussianFromMeanAndPrecision(avgVar, noise)
    let estimateDist = engine.Infer<Gaussian> estVar
    let estimateMean = estimateDist.GetMean()
    let estimateStdDev = Math.Sqrt(estimateDist.GetVariance())
    estimateMean, estimateStdDev

let coinsTest () =
    printf "Sample: %O\n" (coins())
    let (c1D, c2D, bothD) : IDistribution<bool> * IDistribution<bool> * IDistribution<bool> = infer <@ coins @> ()
    (c1D, c2D, bothD)

//let linearRegression () =
//    let nPoints = 10
//    let aTrue, bTrue, invNoiseTrue = LinearRegression.parametersPrior()
//    let data = 
//        [| for x in 1 .. nPoints -> LinearRegression.pointNearLine (float x) aTrue bTrue invNoiseTrue |]
//    printf "true a: %A\n" aTrue
//    printf "true b: %A\n" bTrue
//    printf "true noise (inverse): %A\n" invNoiseTrue
//
//    let (aD: Gaussian), 
//        (bD: Gaussian), 
//        (noiseD: Gamma) = infer <@ LinearRegression.model @> data
//    printf "inferred a: %A\n" aD
//    printf "inferred b: %A\n" bD
//    printf "inferred noise (inverse): %A\n" noiseD
//
//    let aMean = aD.GetMean()
//    let bMean = bD.GetMean()
//    printf "mean a: %A\n" aMean
//    printf "mean b: %A\n" bMean

[<EntryPoint>]
let main argv = 
    let observed = [13.; 15.; 17.; 5.; 20.; 13.; 16.5; 22.;20.;19.;13.;13.;14.]
    let (bD: Gaussian), 
            (noiseD: Gamma) = infer <@gaussianParameters@> observed
    //linearRegression ()
    printfn "%A" argv
    0 // return an integer exit code
