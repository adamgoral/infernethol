using System.Collections.Generic;
using System.ComponentModel;
using MathNet.Numerics.Financial;
using MathNet.Numerics.Statistics;
using SmartTrader.Domain;

namespace InferNetHOL.CSharp.Tests
{
    using System;
    using System.Linq;
    using System.Threading;
    using MicrosoftResearch.Infer.Maths;
    using SmartTrader.Data;

    using MicrosoftResearch.Infer;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Models;

    class Program
    {
        static string[] GetInstruments()
        {
            var instrumentsSource = new LocalInstrumentSource(new Uri(Properties.Settings.Default.instrumentSourcePath));
            return instrumentsSource.GetInstruments(CancellationToken.None).Result.Select(i => i.Symbol).ToArray();
        }

        static DateTime[] GetDates(IEnumerable<IEnumerable<DateTime>> dateGroups)
        {
            var hashSets = dateGroups.Select(ds => new HashSet<DateTime>(ds)).ToArray();
            var union = new HashSet<DateTime>(hashSets.SelectMany(set => set));
            var result = new List<DateTime>();
            foreach (var item in union)
            {
                if (hashSets.All(hs=>hs.Contains(item)))
                {
                    result.Add(item);
                }
            }

            return result.ToArray();
        }

        static Tuple<DateTime, double>[] GetPrices(string instrument)
        {
            var pricesSource = new LocalMarketDataStore(new Uri(Properties.Settings.Default.priceSourcePath));
            var data = pricesSource.GetAsync(new Instrument(instrument), DateTime.MinValue, DateTime.MaxValue,
                CancellationToken.None).Result.ToArray();

            var result = data.Select(d => Tuple.Create(d.Date, d.AdjustedClose)).ToArray();

            return result;
        }

        static Tuple<DateTime, double>[] GetReturns(IEnumerable<Tuple<DateTime, double>> source)
        {
            return
                source.Zip(source.Skip(1), (tuple, tuple1) => Tuple.Create(tuple1.Item1, Math.Log(tuple1.Item2 / tuple.Item2)))
                    .ToArray();
        }

        static double[][] GetMatrix(IEnumerable<IEnumerable<double>> source)
        {
            return source.Select(s => s.ToArray()).ToArray();
        }

        static double[,] GetMatrix(double[][] source)
        {
            var result = new double[source.Length, source[0].Length];
            for (var x = 0; x < result.GetLength(0); x++)
            {
                for (var y = 0; y < result.GetLength(1); y++)
                {
                    result[x, y] = source[x][y];
                }
            }

            return result;
        }

        static double[][] GetReturns(IEnumerable<string> instruments)
        {
            var prices = instruments.Select(i => GetPrices(i).ToDictionary(k => k.Item1, k => k.Item2)).ToArray();
            var dates = new HashSet<DateTime>(GetDates(prices.Select(p => p.Select(kvp => kvp.Key))));
            var filtered =
                prices.Select(
                    p =>
                        p.Where(kvp => dates.Contains(kvp.Key))
                            .Select(kvp => Tuple.Create(kvp.Key, kvp.Value))
                            .OrderBy(k => k.Item1))
                    .ToArray();
            var returns = filtered.Select(i => GetReturns(i).Select(t => t.Item2)).ToArray();
            var result = GetMatrix(returns);
            return result;
        }

        static void TestBasicSetup()
        {
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100);
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1);
            VariableArray<double> data = Variable.Constant(new double[] { 11, 5, 8, 9 });
            Range i = data.Range;
            data[i] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(i);
        }

        static void TestBasicEstimate()
        {
            var observed = new[] { 13.0, 14.0, 15.0, 22.0, 18.0 };
            var evidence = Variable.Bernoulli(0.5);

            // Setup prior distributions
            var avgPrior = Variable.New<Gaussian>();
            var noisePrior = Variable.New<Gamma>();

            // Setup distribution params random variables
            var avg = Variable<double>.Random(avgPrior);
            var noise = Variable<double>.Random(noisePrior);


            var observationSize = Variable.New<int>();
            var observationRange = new Range(observationSize);
            var observedVars = Variable.Array<double>(observationRange);
            using (Variable.ForEach(observationRange))
            {
                observedVars[observationRange] = Variable.GaussianFromMeanAndPrecision(avg, noise);
            }

            avgPrior.ObservedValue = Gaussian.FromMeanAndPrecision(15.0, 0.01);
            noisePrior.ObservedValue = Gamma.FromMeanAndVariance(2, 5);

            observationSize.ObservedValue = observed.Length;
            observedVars.ObservedValue = observed;

            var engine = InferenceEngine.DefaultEngine;
            var avgPosterior = engine.Infer<Gaussian>(avg);
            var noisePosterior = engine.Infer<Gamma>(noise);
            var estimate = Variable.GaussianFromMeanAndPrecision(avg, noise);
            var estimateDist = engine.Infer<Gaussian>(estimate);
            var probUnderX = engine.Infer<Bernoulli>(estimate < 22.0).GetProbTrue();
        }

        private static void LinearRegressionTest(bool useLoop, int observationSize, double aActual, double bActual)
        {
            var rand = new System.Random();
            var actuals =
                Enumerable.Range(0, observationSize)
                    .Select(i => Tuple.Create(i, bActual*i + aActual + rand.NextDouble()))
                    .ToArray();

            var aPrior = Variable.GaussianFromMeanAndPrecision(0, 0.01).Named("aPrior");
            var bPrior = Variable.GaussianFromMeanAndPrecision(0, 0.01).Named("bPrior");
            var noisePrior = Variable.GaussianFromMeanAndPrecision(0, 0.01).Named("noisePrior");
            var obsRange = new Range(actuals.Length);
            var xArray = Variable.Array<double>(obsRange);
            var exprArray = Variable.Array<double>(obsRange);
            if (useLoop)
            {
                foreach (var actual in actuals)
                {
                    var x = Variable.New<double>();
                    var expr = aPrior + bPrior * x + noisePrior;
                    expr.ObservedValue = actual.Item2;
                    x.ObservedValue = actual.Item1;
                }
            }
            else
            {
                using (Variable.ForEach(obsRange))
                {
                    exprArray[obsRange] = aPrior + xArray[obsRange] * bPrior + noisePrior;
                }

                xArray.ObservedValue = actuals.Select(t => (double) t.Item1).ToArray();
                exprArray.ObservedValue = actuals.Select(t => t.Item2).ToArray();
            }

            var engine = new InferenceEngine();
            var aPosterior = engine.Infer<Gaussian>(aPrior);
            var bPosterior = engine.Infer<Gaussian>(bPrior);
            var noisePosterior = engine.Infer<Gaussian>(noisePrior);
            Console.WriteLine("aPosterior: {0}", aPosterior);
            Console.WriteLine("bPosterior: {0}", bPosterior);
            Console.WriteLine("noisePosterior: {0}", noisePosterior);
        }

        private static void GeneralLinearModel(int rows, int cols)
        {
            var rand = new System.Random();
            var k = new Range(cols).Named("k");
            var l = new Range(rows).Named("l");
            var X = new double[rows,cols];
            var Y = new double[rows];
            for (var y = 0; y < rows; y++)
            {
                Y[y] = y;
                for (var x = 0; x < cols; x++)
                {
                    X[y,x] = (x + 1) * y;
                }
            }

            var meansPrior = Variable.Array<double>(k);
            using (Variable.ForEach(k))
            {
                meansPrior[k] = Variable.GaussianFromMeanAndPrecision(0, 0.01);                
            }

            var noisePrior = Variable.GaussianFromMeanAndPrecision(0.0, 0.01);
            var xV = Variable.Array<double>(l, k); 
            var yA = Variable.Array<double>(l);
            using (Variable.ForEach(l))
            {
                var elems = Variable.Array<double>(k);
                using (Variable.ForEach(k))
                {
                    elems[k] = meansPrior[k] * xV[l, k];
                }

                yA[l] = Variable.Sum(elems) + noisePrior;
            }

            xV.ObservedValue = X;
            yA.ObservedValue = Y;
            var engine = new InferenceEngine();
            var meansPosterior = engine.Infer<DistributionStructArray<Gaussian, double>>(meansPrior);
            var noisePosterior = engine.Infer<Gaussian>(noisePrior);
        }

        private static void VectorSampling()
        {
            var sampleNumber = 100;
            var cols = 2;
            var rows = cols;
            var means = Enumerable.Range(0, cols).Select(_ => 0.0).ToArray();
            var cov = new double[rows, cols];
            for (var c = 0; c < cols; c++)
            {
                for (var r = 0; r < rows; r++)
                {
                    if (r==c)
                    {
                        cov[r, c] = 1;
                    }
                    else
                    {
                        cov[r, c] = 0.5;
                    }
                }
            }

            var vector = VectorGaussian.FromMeanAndPrecision(Vector.FromArray(means),
                new PositiveDefiniteMatrix(cov));
            Vector.FromArray(new double[] {0.1, 0.1});
            var samples = Enumerable.Range(0, sampleNumber).Select(_ => vector.Sample()).ToArray();
        }

        static IEnumerable<double[]> Samples(VectorGaussian distribution)
        {
            while (true)
            {
                yield return distribution.Sample().ToArray();
            }
        }

        private static void PortfolioWeightsMultivariateTest()
        {
            Console.WriteLine("Simple portfolio weights inference example\n");
            var instrumentCount = 10;
            var instruments = GetInstruments().Where(i => GetPrices(i).Length > 1000).Take(instrumentCount).ToArray();
            // Annualised returns
            var returnsMatrix = GetReturns(instruments).Select(d => d.Select(i => i * 250.0).ToArray()).ToArray();
            PortfolioWeightsMultivariate(returnsMatrix);
        }

        private static void PortfolioWeightsMultivariate(double[][] obs)
        {
            // Setup input vector array
            var r = new Range(obs[0].Length).Named("r");
            var c = new Range(obs.Length).Named("c");
            var data = Variable.Array<Vector>(r).Named("matrix");

            // Setup multivariate distribution priors
            var meansPrior = Variable.VectorGaussianFromMeanAndPrecision(Vector.FromArray(new double[obs.Length]), PositiveDefiniteMatrix.IdentityScaledBy(obs.Length, 10));
            var covPrior = Variable.WishartFromShapeAndScale(1, PositiveDefiniteMatrix.Identity(c.SizeAsInt));
            using (Variable.ForEach(r))
            {
                data[r] = Variable.VectorGaussianFromMeanAndPrecision(meansPrior, covPrior);
            }
            
            // Observe
            data.ObservedValue = GetRows(obs).Select(Vector.FromArray).ToArray();

            var engine = new InferenceEngine(new VariationalMessagePassing());
            engine.ShowProgress = false;

            // Infer posterior means
            var meansPosterior = engine.Infer<VectorGaussian>(meansPrior).GetMean();
            Console.WriteLine("Inferred means {0}\n", meansPosterior);
            // Infer posterior covariance
            var covPosterior = engine.Infer<Wishart>(covPrior);
            Console.WriteLine("Inferred Covariance\n{0}", covPosterior.GetMean());

            // Setup instruments vector gaussian variable
            var instruments = Variable.VectorGaussianFromMeanAndPrecision(meansPosterior, covPosterior.GetMean());

            // Setup weights vector with 0 means and identity covariance
            var weights = Variable.VectorGaussianFromMeanAndPrecision(Vector.FromArray(new double[obs.Length]), PositiveDefiniteMatrix.Identity(obs.Length));
            
            // Setup porfolio variable as a product or instruments and weights
            var portfolio = Variable.InnerProduct(instruments, weights);
            for (var i = 0.1; i < 0.5; i += 0.1)
            {
                // Constrain portfolio to specified gaussian distribution
                var constraintDistribution = new Gaussian(i, 0.001);
                Variable.ConstrainEqualRandom(portfolio, constraintDistribution);
                // Infer posterior weights
                var posteriorWeights = engine.Infer<VectorGaussian>(weights);
                Console.WriteLine("\nGiven portfolio constraint {0}", constraintDistribution);
                Console.WriteLine("inferred weights are\t{0}", posteriorWeights.GetMean());
                var newPortfolioPosterior = engine.Infer<Gaussian>(Variable.InnerProduct(instruments, Variable.Random(posteriorWeights)));
                Console.WriteLine("portfolio return mean {0} variance {1}", newPortfolioPosterior.GetMean(), newPortfolioPosterior.GetVariance());
            }
        }

        private static void TrainCovariance(double[][] obs)
        {
            var r = new Range(obs[0].Length).Named("r");
            var c = new Range(obs.Length).Named("c");
            var data = Variable.Array<Vector>(r).Named("matrix");

            var meansPrior = Variable.VectorGaussianFromMeanAndPrecision(Vector.FromArray(new double[obs.Length]), PositiveDefiniteMatrix.IdentityScaledBy(obs.Length, 10));
            var covPrior = Variable.WishartFromShapeAndScale(1, PositiveDefiniteMatrix.Identity(c.SizeAsInt));
            using (Variable.ForEach(r))
            {
                data[r] = Variable.VectorGaussianFromMeanAndPrecision(meansPrior, covPrior);
            }

            data.ObservedValue = GetRows(obs).Select(Vector.FromArray).ToArray();

            var engine = new InferenceEngine(new VariationalMessagePassing());
            
            var meansPosterior = engine.Infer<VectorGaussian>(meansPrior);
            
            var covPosterior = engine.Infer<Wishart>(covPrior);
            var covPosteriorVariance = covPosterior.GetMean().Inverse();

            var verificationGaussian = new VectorGaussian(meansPosterior.GetMean(), covPosteriorVariance);
            var verificationSamples = GetRows(Samples(verificationGaussian).Take(1000).ToArray()).ToArray();
            var covM = GetCovarianceMatrix(verificationSamples);
            var verificationCovariance = covM.GetVariance();

            var actual = GetCovarianceMatrix(obs);
            var actualMeans = actual.GetMean();
            var actualCov = actual.GetVariance();
        }

        private static Gaussian InferGaussian(double[] data)
        {
            return InferGaussian(data, new InferenceEngine(new VariationalMessagePassing()));// InferenceEngine.DefaultEngine);
        }

        private static Gaussian InferGaussian(double[] data, InferenceEngine engine)
        {
            var r = new Range(data.Length);
            var meanPrior = Variable.GaussianFromMeanAndPrecision(0, 0.01);
            var precPrior = Variable.GammaFromShapeAndScale(1, 1);
            var xs = Variable.Array<double>(r);
            using (Variable.ForEach(r))
            {
                xs[r] = Variable.GaussianFromMeanAndPrecision(meanPrior, precPrior);
            }

            xs.ObservedValue = data;
            var estimateDist = engine.Infer<Gaussian>(Variable.GaussianFromMeanAndPrecision(meanPrior, precPrior));
            return estimateDist;
        }

        private static void PortfolioModel(double[] marketData, double[] goldData, double[][] stocksData)
        {
            var vmpEngine = new InferenceEngine(new VariationalMessagePassing());
            vmpEngine.ShowProgress = false;
            // Ranges
            var r = new Range(marketData.Length).Named("r");
            var c = new Range(stocksData.Length).Named("c");

            // Infer observed data distributions
            var marketDistribution = InferGaussian(marketData, vmpEngine);
            Console.WriteLine("Market return distribution {0}", marketDistribution);
            var goldDistribution = InferGaussian(goldData, vmpEngine);
            Console.WriteLine("Gold return distribution {0}", goldDistribution);
            var stocksDistribution = stocksData.Select(d => InferGaussian(d, vmpEngine)).ToArray();
            Console.WriteLine("Stock return distributions {0}\n", string.Join(" ", stocksDistribution.Select(s => s.ToString())));

            // declare supporting variables for market weight inference
            var marketV = Variable.Array<double>(r).Named("marketV");
            var marketWeights = Variable.Array<double>(c).Named("marketWeights");

            // declare supporting variables for market weight inference
            var goldV = Variable.Array<double>(r).Named("goldV");
            var goldWeights = Variable.Array<double>(c).Named("goldWeights");

            // declare supporting variables for coefficients & noise inference
            var coefficientNoisePrior = Variable.Array<double>(c).Named("coefficientNoise");
            var coefficientIntercept = Variable.Array<double>(c).Named("coefficientIntercept");
            var stocksV = Variable.Array<double>(c, r).Named("stocksV");

            // Stage 1 - infer stock coefficients and distributions
            using (Variable.ForEach(c))
            {
                marketWeights[c] = Variable.GaussianFromMeanAndPrecision(0, 0.01);
                goldWeights[c] = Variable.GaussianFromMeanAndPrecision(0, 0.01);
                coefficientNoisePrior[c] = Variable.GammaFromShapeAndScale(1, 1);
                coefficientIntercept[c] = Variable.GaussianFromMeanAndPrecision(0, 0.01);

                using (Variable.ForEach(r))
                {
                    stocksV[c, r] = Variable.GaussianFromMeanAndPrecision(marketV[r] * marketWeights[c] + goldV[r] * goldWeights[c] + coefficientIntercept[c], coefficientNoisePrior[c]);
                }
            }

            // Observation
            goldV.ObservedValue = goldData;
            marketV.ObservedValue = marketData;
            stocksV.ObservedValue = GetMatrix(stocksData);

            // Inference
            var marketWPosterior = vmpEngine.Infer<DistributionStructArray<Gaussian, double>>(marketWeights);
            Console.WriteLine("Market coefficient weights\t{0}", string.Join(" ", marketWPosterior.Select(s => s.GetMean())));
            var goldWPosterior = vmpEngine.Infer<DistributionStructArray<Gaussian, double>>(goldWeights);
            Console.WriteLine("Gold coefficient weights\t{0}", string.Join(" ", goldWPosterior.Select(s => s.GetMean())));
            var coefficientInterceptPosterior = vmpEngine.Infer<DistributionStructArray<Gaussian, double>>(coefficientIntercept);
            Console.WriteLine("Coefficient intercept\t\t{0}", string.Join(" ", coefficientInterceptPosterior.Select(s => s.GetMean())));
            var coefficientNoisePosterior = vmpEngine.Infer<DistributionStructArray<Gamma, double>>(coefficientNoisePrior);
            Console.WriteLine("Coefficient precision\t\t{0}\n", string.Join(" ", coefficientNoisePosterior.Select(s => s.GetMean())));












            // Stage 2 - infer portfolio weights
            var portfolioW = Variable.Array<double>(c).Named("portfolioW");
            var portfolio = Variable.New<double>().Named("portfolio");
            var portfolioNoise = Variable.GammaFromShapeAndScale(1, 5).Named("portfolioNoise");
            var sumProd = Variable.Array<double>(c);
            var stockMeans = Variable.Array<double>(c).Named("stockMeans");

            // Create stock random variables based on the inferred posterior
            for (var i = 0; i < c.SizeAsInt; i++)
            {
                stockMeans[i] = Variable.Random(stocksDistribution[i]);
            }

            using (var ci = Variable.ForEach(c))
            {
                portfolioW[c] = Variable.GaussianFromMeanAndPrecision(0, 0.01);
                sumProd[c] = stockMeans[c] * portfolioW[c];
            }

            portfolio = Variable.Sum(sumProd);

            // Set distribution constraint
            Variable.ConstrainEqualRandom(portfolio, new Gaussian(0.1, 0.01));
            // Infer portfolio weights
            var portfolioWPosterior = vmpEngine.Infer<DistributionStructArray<Gaussian, double>>(portfolioW);
            Console.WriteLine("Portfolio weights\t\t{0}\n", string.Join(" ", portfolioWPosterior.Select(s => s.GetMean())));











            // Stage 3 - setup model using inferred weights
            // Declare model variables
            var goldVar = Variable.Random(goldDistribution);
            var marketVar = Variable.Random(marketDistribution);
            var stocksVars = Variable.Array<double>(c);
            var goldWVar = Variable.Array<double>(c);
            var marketWVar = Variable.Array<double>(c);
            var portfolioWVar = Variable.Array<double>(c);
            var coefficientNoiseVar = Variable.Array<double>(c);
            var coefficientInterceptVar = Variable.Array<double>(c);

            // Set observed weights and coefficients
            goldWVar.ObservedValue = goldWPosterior.Select(p => p.GetMean()).ToArray();
            marketWVar.ObservedValue = marketWPosterior.Select(p => p.GetMean()).ToArray();
            portfolioWVar.ObservedValue = portfolioWPosterior.Select(p => p.GetMean()).ToArray();
            coefficientNoiseVar.ObservedValue=coefficientNoisePosterior.Select(p=>p.GetMean()).ToArray();
            coefficientInterceptVar.ObservedValue = coefficientInterceptPosterior.Select(p => p.GetMean()).ToArray();

            var prod = Variable.Array<double>(c);
            using (Variable.ForEach(c))
            {
                stocksVars[c] = Variable.GaussianFromMeanAndPrecision(goldVar * goldWVar[c] + marketVar * marketWVar[c] + coefficientInterceptVar[c], coefficientNoiseVar[c]);
                prod[c] = stocksVars[c] * portfolioWVar[c];
            }

            var portfolioVar = Variable.Sum(prod);
            var inferredPortfolio = vmpEngine.Infer<Gaussian>(portfolioVar);
            Console.WriteLine("Given market return {0} and gold return {1},  inferred portfolio return is {2}\n", marketDistribution.GetMean(), goldDistribution.GetMean(), inferredPortfolio.GetMean());

            Console.WriteLine("What-if analysis:\n");
            // Set market assumptions
            for (var i = 0.1; i < 0.5; i += 0.1)
            {
                marketVar.ObservedValue = i;
                Console.WriteLine("Given market return {0} and gold return {1},", i, goldDistribution.GetMean());
                // infer stock returns
                var infferredStockVars = vmpEngine.Infer<DistributionStructArray<Gaussian, double>>(stocksVars);
                Console.WriteLine("inferred stock returns are\t\t{0}", string.Join(" ", infferredStockVars.Select(p => p.GetMean())));
                // infer portfolio
                inferredPortfolio = vmpEngine.Infer<Gaussian>(portfolioVar);
                Console.WriteLine("inferred portfolio return is\t\t{0}\n", inferredPortfolio.GetMean());
            }

            marketVar.ClearObservedValue();

            // Set gold assumptions
            for (var i = 0.1; i < 0.5; i += 0.1)
            {
                goldVar.ObservedValue = i;
                Console.WriteLine("Given market return {0} and gold return {1},", marketDistribution.GetMean(), i);
                var infferredStockVars = vmpEngine.Infer<DistributionStructArray<Gaussian, double>>(stocksVars);
                Console.WriteLine("inferred stock returns are\t\t{0}", string.Join(" ", infferredStockVars.Select(p => p.GetMean())));
                // infer portfolio
                inferredPortfolio = vmpEngine.Infer<Gaussian>(portfolioVar);
                Console.WriteLine("inferred portfolio return is\t\t{0}\n", inferredPortfolio.GetMean());
            }

            goldVar.ClearObservedValue();
        }

        private static void PortfolioModelTest()
        {
            Console.WriteLine("Simple portfolio model example\n");
            var instruments = new[] { "GLD", "SPY", "ABX", "AGD.L", "GOLD" };
            var prices = instruments.Where(i => GetPrices(i).Length > 1).ToArray();
            var returnsMatrix = GetReturns(prices).Select(d => d.Select(i => i * 250).ToArray()).ToArray();
            PortfolioModel(returnsMatrix[1], returnsMatrix[0], returnsMatrix.Skip(2).ToArray());
        }

        private static IEnumerable<double[]> GetRows(double[][] source)
        {
            var cols = source.Length;
            var rows = source[0].Length;

            for (var r = 0; r < rows; r++)
            {
                var result = new double[cols];
                for (var c = 0; c < cols; c++)
                {
                    result[c] = source[c][r];
                }

                yield return result;
            }
        }

        private static VectorGaussian GetCovarianceMatrix(double[][] data)
        {
            var rows = data[0].Length;
            var columns = data.Length;
            var means = data.Select(d => d.Average()).ToArray();
            var cov = new double[columns, columns];

            for (var x = 0; x < columns; x++)
            {
                for (var y = 0; y < columns; y++)
                {
                    cov[x, y] = Covariance(data[x], data[y]);
                }
            }

            var result = new VectorGaussian(Vector.FromArray(means), new PositiveDefiniteMatrix(cov));
            return result;
        }

        private static double Covariance(IEnumerable<double> xs, IEnumerable<double> ys)
        {
            var xAvg = xs.Average();
            var yAvg = ys.Average();
            var cov = xs.Zip(ys, (x, y) => (x - xAvg) * (y - yAvg)).Average();
            return cov;
        }

        private static void SamplingTest()
        {
            //VectorSampling();

            var instrumentCount = 10;
            var instruments = GetInstruments().Where(i => GetPrices(i).Length > 1000).Take(instrumentCount).ToArray();
            var returnsMatrix = GetReturns(instruments);
            //TrainCorrelation(returnsMatrix);
            var samples = Samples(GetCovarianceMatrix(returnsMatrix)).Take(5000).ToArray();
            for (var i = 0; i < instrumentCount; i++)
            {
                var index = i;
                var stdDevSource = returnsMatrix[index].StandardDeviation();
                var stdDevSample = samples.Select(v => v[index]).StandardDeviation();
                Console.WriteLine("{0} stdev: {1} stddev of sampling: {2}", instruments[index], stdDevSource,
                    stdDevSample);
                for (var j = 0; j < instrumentCount; j++)
                {
                    if (i != j)
                    {
                        var covSource = Covariance(returnsMatrix[i], returnsMatrix[j]);
                        var covSample = Covariance(samples.Select(v => v[i]), samples.Select(v => v[j]));
                        Console.WriteLine("{0} {1} cov: {2} cov of sampling {3}", instruments[i], instruments[j],
                            covSource, covSample);
                    }
                }
            }
        }

        private static void FittingCurveWithBayesian(double[] input, double[] observed, int m)
        {
            if (input.Length != observed.Length)
            {
                throw new ArgumentException("input and observed arrays must be equal in length");
            }

            var rR = new Range(input.Length).Named("r");
            var rM = new Range(m).Named("M");

            var inputArr = new double[input.Length, m];
            for (var r = 0; r < input.Length; r++)
            {
                for (var c = 0; c < m; c++)
                {
                    inputArr[r, c] = Math.Pow(input[r], c);
                }
            }

            var X = Variable.Array<double>(rR, rM).Named("X");
            X.ObservedValue = inputArr;

            var w = Variable.Array<double>(rM).Named("W");
            w[rM] = Variable.GaussianFromMeanAndPrecision(0, 0.01).ForEach(rM);

            var y = Variable.Array<double>(rR).Named("Y");
            using (Variable.ForEach(rR))
            {
                var prods = Variable.Array<double>(rM);
                using (Variable.ForEach(rM))
                {
                    prods[rM] = X[rR, rM]*w[rM];
                }

                y[rR] = Variable.Sum(prods);
            }

            y.ObservedValue = observed;

            var engine = new InferenceEngine();
            var posteriorW = engine.Infer<DistributionStructArray<Gaussian, double>>(w);
            Console.WriteLine("{0} deg poli weights\n{1}", m, posteriorW);
        }

        private static void TrainGaussian(double sourceMean, double sourcePrecision, int cycles)
        {
            var source = Gaussian.FromMeanAndPrecision(sourceMean, sourcePrecision);

            // Prior distributions
            var meanPriorDistr = Gaussian.FromMeanAndPrecision(0, 0.01);
            var precisionPriorDistr = Gamma.FromMeanAndVariance(2, 5);

            var meanPrior = Variable.Random(meanPriorDistr).Named("mean");
            var precPrior = Variable.Random(precisionPriorDistr).Named("precision");

            var engine = new InferenceEngine();
            for (var i = 0; i < cycles; i++)
            {
                var x = Variable.GaussianFromMeanAndPrecision(meanPrior, precPrior);
                x.ObservedValue = source.Sample();
            }

            var meanPost = engine.Infer<Gaussian>(meanPrior);
            var precPost = engine.Infer<Gamma>(precPrior);
            var estimate = Variable.GaussianFromMeanAndPrecision(meanPrior, precPrior);
            var estimateDist = engine.Infer<Gaussian>(estimate);
            Console.WriteLine("mean: {0}, prec: {1}", estimateDist.GetMean(), estimateDist.Precision);
        }

        [STAThread]
        private static void Main(string[] args)
        {
            PortfolioWeightsMultivariateTest();
            Console.ReadLine();
            Console.Clear();
            PortfolioModelTest();
            //DemoMultivariateGaussianTraining();
            //TrainCorrelationTest();
            //TrainGaussian(5, 0.8, 40);
            //CurveFittingTest();
            //GeneralLinearModel(100,3);
            //LinearRegressionTest(true, 100, 10.0, 4.0);
            //LinearRegressionTest(false, 100, 10.0, 4.0);
            Console.ReadLine();
        }

        private static void TrainCorrelationTest()
        {
            var instrumentCount = 10;
            var instruments = GetInstruments().Where(i => GetPrices(i).Length > 1000).Take(instrumentCount).ToArray();
            var returnsMatrix = GetReturns(instruments);
            TrainCovariance(returnsMatrix);
        }

        private static void CurveFittingTest()
        {
            var input = Enumerable.Range(1, 100).Select(x => Convert.ToDouble(x)/100.0).ToArray();
            var expected = input.Select(x => Math.Sin(2*Math.PI*x)).ToArray();
            for (var i = 2; i < 10; i++)
            {
                FittingCurveWithBayesian(input, expected, i);
            }
        }

        public static Tuple<VectorGaussian, Wishart> LearnGaussian(Vector[] obs)
        {
            int numData = obs.Length;
            int dim = obs[0].Count;
            Variable<Vector> mean = Variable.VectorGaussianFromMeanAndPrecision(
                Vector.Zero(dim),
                PositiveDefiniteMatrix.IdentityScaledBy(dim, 10.0)).Named("mean");
            Variable<PositiveDefiniteMatrix> prec = Variable.WishartFromShapeAndScale(
                100.0, PositiveDefiniteMatrix.IdentityScaledBy(dim, 0.01));
            Range n = new Range(obs.Length).Named("n");
            VariableArray<Vector> data = Variable.Array<Vector>(n).Named("x");
            data[n] = Variable.VectorGaussianFromMeanAndPrecision(mean, prec).ForEach(n);
            data.ObservedValue = obs;

            var engine = new InferenceEngine(new VariationalMessagePassing());
            var meanPosterior = engine.Infer<VectorGaussian>(mean);
            var precPosterior = engine.Infer<Wishart>(prec);

            return new Tuple<VectorGaussian, Wishart>(meanPosterior, precPosterior);
        }

        public static VectorGaussian Conditional(Tuple<VectorGaussian, Wishart> priors, int observedIndex, double observedValue)
        {
            Variable<Vector> mean = Variable.Random(priors.Item1);
            Variable<PositiveDefiniteMatrix> prec = Variable.Random(priors.Item2);
            Variable<Vector> v = Variable.VectorGaussianFromMeanAndPrecision(mean, prec);
            // Initialise v to a proper distribution (to avoid improper messages)
            v.InitialiseTo(new VectorGaussian(priors.Item1.GetMean(), priors.Item2.GetMean()));
            Variable<double> observedV = Variable.GetItem(v, observedIndex);
            observedV.ObservedValue = observedValue;
            var engine = new InferenceEngine(new VariationalMessagePassing());
            var vPosterior = engine.Infer<VectorGaussian>(v);
            return vPosterior;
        }

        static void DemoMultivariateGaussianTraining()
        {
            Vector[] obs = new Vector[] {
				Vector.FromArray(1.0, -2.0),
				Vector.FromArray(1.5, 1.0),
				Vector.FromArray(2.0, -1.0),
				Vector.FromArray(0.5, 2.5),
				Vector.FromArray(0, -1)};
            var priors = LearnGaussian(obs);
            Console.WriteLine(priors.Item1);
            double val = -2.0;
            double inc = .5;
            for (int i = 0; i < 9; i++)
            {
                Console.WriteLine(Conditional(priors, 0, val));
                val += inc;
            }
        }
    }
}
