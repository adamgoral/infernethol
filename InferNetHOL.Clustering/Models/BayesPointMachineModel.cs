using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InferNetHOL.Clustering.Models
{
    public class BayesPointMachineModel
    {
        //public int[] Classification(Vector[] features, int[] labels)
        //{

        //}

        public double[] Regression(Vector[] features, double[] values)
        {
            var wMeans = Variable.Vector(Vector.Zero(features[0].Count).ToArray());
            var wPrecision = Variable.WishartFromShapeAndRate(100, PositiveDefiniteMatrix.IdentityScaledBy(features[0].Count, 0.01));
            var w = Variable.VectorGaussianFromMeanAndPrecision(wMeans, wPrecision).Named("w");
            var numItems = Variable.New<int>().Named("numItems");
            var i = new Range(numItems).Named("i");
            i.AddAttribute(new Sequential());

            var noisePrecision = Variable.New<double>().Named("noisePrecision");

            var x = Variable.Array<Vector>(i).Named("x");
            var y = Variable.Array<double>(i).Named("y");

            using (Variable.ForEach(i))
            {
                y[i] = Variable.GaussianFromMeanAndPrecision(Variable.InnerProduct(w, x[i]), noisePrecision);
            }

            numItems.ObservedValue = features.Length;
            x.ObservedValue = features;
            y.ObservedValue = values;

            var engine = new InferenceEngine();
            engine.Compiler.UseSerialSchedules = true;
            engine.ShowProgress = false;
            var wPosterior = engine.Infer<VectorGaussian>(w);
            y.ClearObservedValue();
            w.ObservedValue = wPosterior.GetMean();
            var inferredValues = engine.Infer<IList<Gaussian>>(y);
            return inferredValues.Select(v => v.GetMean()).ToArray();
        }

        public InferenceResult<ClusterRegressionWeights[]> Regression(Vector[] features, double[] values, int clusters)
        {
            var dimensions = features[0].Count;
            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            var evidenceBlock = Variable.If(evidence);

            var clustersRange = new Range(clusters).Named("clustersRange");
            var wMeans = Variable.Array<Vector>(clustersRange).Named("wMeans");
            var wPrecision = Variable.Array<PositiveDefiniteMatrix>(clustersRange).Named("wPrecision");
            var noise = Variable.Array<double>(clustersRange).Named("noise");
            var w = Variable.Array<Vector>(clustersRange).Named("w");
            using(Variable.ForEach(clustersRange))
            {
                wMeans[clustersRange] = Variable
                    .VectorGaussianFromMeanAndPrecision(
                    Vector.Zero(dimensions),
                    PositiveDefiniteMatrix.IdentityScaledBy(dimensions, 0.01));
                wPrecision[clustersRange] = Variable.WishartFromShapeAndRate(100, PositiveDefiniteMatrix.IdentityScaledBy(dimensions, 0.01));
                w[clustersRange] = Variable.VectorGaussianFromMeanAndPrecision(wMeans[clustersRange], wPrecision[clustersRange]);
                noise[clustersRange] = Variable.GammaFromMeanAndVariance(2, 5);
            }
             
            var numItems = Variable.New<int>().Named("numItems");
            var i = new Range(numItems).Named("i");
            i.AddAttribute(new Sequential());

            var initialWeights = Enumerable.Range(0, clusters).Select(_ => 1.0).ToArray();
            var mixtureWeightsPrior = Variable.Dirichlet(clustersRange, initialWeights).Named("mixtureWeightsPrior");

            var latentIndex = Variable.Array<int>(i).Named("latentIndex");

            var x = Variable.Array<Vector>(i).Named("x");
            var y = Variable.Array<double>(i).Named("y");

            using (Variable.ForEach(i))
            {
                latentIndex[i] = Variable.Discrete(mixtureWeightsPrior);
                using(Variable.Switch(latentIndex[i]))
                {
                    y[i] = Variable.GaussianFromMeanAndPrecision(Variable.InnerProduct(w[latentIndex[i]], x[i]), noise[latentIndex[i]]);
                }
            }

            numItems.ObservedValue = features.Length;
            var zinit = new Discrete[features.Length];
            for (var j = 0; j < zinit.Length; j++)
                zinit[j] = Discrete.PointMass(Rand.Int(clustersRange.SizeAsInt), clustersRange.SizeAsInt);
            latentIndex.InitialiseTo(Distribution<int>.Array(zinit));

            evidenceBlock.CloseBlock();

            x.ObservedValue = features;
            y.ObservedValue = values;

            var engine = new InferenceEngine(new VariationalMessagePassing());
            engine.Compiler.UseSerialSchedules = true;
            engine.ShowProgress = false;
            var wPosterior = engine.Infer<IList<VectorGaussian>>(w);
            var mixtureWeightsPosterior = engine.Infer<Dirichlet>(mixtureWeightsPrior);
            var noisePosterior = engine.Infer<IList<Gamma>>(noise);
            var bEvidence = engine.Infer<Bernoulli>(evidence);

            var clusterList = new List<ClusterRegressionWeights>();
            for (var c = 0; c < clusters; c++ )
            {
                clusterList.Add(new ClusterRegressionWeights(wPosterior[c].GetMean(), noisePosterior[c], mixtureWeightsPosterior.GetMean()[c]));
            }

            return new InferenceResult<ClusterRegressionWeights[]>(bEvidence, clusterList.ToArray());
        }

        public class ClusterRegressionWeights
        {
            public ClusterRegressionWeights(Vector weights, Gamma precision, double clusterWeight)
            {
                this.Weights = weights;
                this.Precision = precision;
                this.ClusterWeight = clusterWeight;
            }

            public Vector Weights { get; private set; }

            public Gamma Precision { get; private set; }

            public double ClusterWeight { get; private set; }

            public override string ToString()
            {
                return string.Format("Cluster mix: {0} precision: {1} weights: [{2}]", this.ClusterWeight, this.Precision.GetMean(), string.Join(",", this.Weights.ToArray()));
            }
        }
    }
}
