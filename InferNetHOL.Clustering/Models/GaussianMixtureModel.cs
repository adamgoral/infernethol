namespace InferNetHOL.Clustering.Models
{
    using MicrosoftResearch.Infer;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;
    using MicrosoftResearch.Infer.Models;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    public class GaussianMixtureModel
    {
        /// <summary>
        /// Infer without using Infer.Net
        /// </summary>
        /// <param name="observedData">Array of observed data</param>
        /// <param name="clusters">Assumed cluster count</param>
        /// <returns></returns>
        public InferenceResult<Cluster[]> InferStandard(Vector[] observedData, int clusters)
        {
            var mixture = new Accord.MachineLearning.GaussianMixtureModel(clusters);
            var logLikelihood = mixture.Compute(observedData.Select(d => d.ToArray()).ToArray());
            var distribution = mixture.ToMixtureDistribution();
            var result = new List<Cluster>();
            foreach(var component in distribution.Components)
            {
                result.Add(new Cluster(Vector.FromArray(component.Mean), new PositiveDefiniteMatrix(component.Covariance)));
            }

            return new InferenceResult<Cluster[]>(Bernoulli.FromLogOdds(logLikelihood), result.ToArray());
        }

        public InferenceResult<Cluster[]> Infer(Vector[] observedData, int clusters)
        {
            var dimensions = observedData.First().Count;
            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            var evidenceBlock = Variable.If(evidence);
            var clustersRange = new Range(clusters).Named("clustersRange");
            var meansPrior = Variable.Array<Vector>(clustersRange).Named("meansPrior");
            meansPrior[clustersRange] = Variable
                .VectorGaussianFromMeanAndPrecision(
                    Vector.Zero(dimensions),
                    PositiveDefiniteMatrix.IdentityScaledBy(dimensions, 0.01))
                .ForEach(clustersRange);

            var precisionsPrior = Variable.Array<PositiveDefiniteMatrix>(clustersRange).Named("precisionsPrior");
            precisionsPrior[clustersRange] = Variable.WishartFromShapeAndRate(100, PositiveDefiniteMatrix.IdentityScaledBy(dimensions, 0.01))
                .ForEach(clustersRange);

            var initialWeights = Enumerable.Range(0, clusters).Select(_ => 1.0).ToArray();
            var mixtureWeightsPrior = Variable.Dirichlet(clustersRange, initialWeights).Named("mixtureWeightsPrior");

            var dataRange = new Range(observedData.Length).Named("dataRange");
            var data = Variable.Array<Vector>(dataRange).Named("data");

            var latentIndex = Variable.Array<int>(dataRange).Named("latentIndex");

            using (Variable.ForEach(dataRange))
            {
                latentIndex[dataRange] = Variable.Discrete(mixtureWeightsPrior);
                using (Variable.Switch(latentIndex[dataRange]))
                {
                    data[dataRange] = Variable.VectorGaussianFromMeanAndPrecision(meansPrior[latentIndex[dataRange]], precisionsPrior[latentIndex[dataRange]]);
                }
            }

            var zinit = new Discrete[dataRange.SizeAsInt];
            for (int i = 0; i < zinit.Length; i++)
                zinit[i] = Discrete.PointMass(Rand.Int(clustersRange.SizeAsInt), clustersRange.SizeAsInt);
            latentIndex.InitialiseTo(Distribution<int>.Array(zinit));

            evidenceBlock.CloseBlock();

            data.ObservedValue = observedData;

            var ie = new InferenceEngine(new VariationalMessagePassing());
            ie.ShowProgress = false;

            var mixtureWeightsPosterior = ie.Infer(mixtureWeightsPrior);
            var meansPosterior = ie.Infer<VectorGaussian[]>(meansPrior);
            var precisionsPosterior = ie.Infer<Wishart[]>(precisionsPrior);
            var bEvidence = ie.Infer<Bernoulli>(evidence);

            var result = new List<Cluster>();
            for (var i = 0; i < clusters; i++)
            {
                result.Add(new Cluster(meansPosterior[i].GetMean(), precisionsPosterior[i].GetMean().Inverse()));
            }

            return new InferenceResult<Cluster[]>(bEvidence, result.ToArray());
        }
    }

    public class InferenceResult<T>
    {
        public InferenceResult(Bernoulli evidence, T result)
        {
            this.Evidence = evidence;
            this.Result = result;
        }

        public Bernoulli Evidence { get; private set; }

        public T Result { get; private set; }
    }

    public class Cluster
    {
        public Cluster(Vector means, PositiveDefiniteMatrix covariance)
        {
            this.Means = means;
            this.Covariance = covariance;
        }

        public Vector Means { get; private set; }

        public PositiveDefiniteMatrix Covariance { get; private set; }
    }
}
