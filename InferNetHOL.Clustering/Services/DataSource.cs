namespace InferNetHOL.Clustering.Services
{
    using MicrosoftResearch.Infer.Maths;
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    public class DataSource
    {
        private readonly Uri dataPath;

        public DataSource(Uri dataPath)
        {
            this.dataPath = dataPath;
        }

        public string[] GetColumns(string dataSetName)
        {
            var path = new Uri(dataPath, dataSetName);
            using (var reader = new StreamReader(File.OpenRead(path.AbsolutePath)))
            {
                var parts = new string[0];
                if (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    parts = line.Split(',');
                }

                return parts;
            }
        }

        public IEnumerable<Vector> Load(string dataSetName)
        {
            var path = new Uri(dataPath, dataSetName);
            using (var reader = new StreamReader(File.OpenRead(path.AbsolutePath)))
            {
                if (!reader.EndOfStream)
                    reader.ReadLine();
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var parts = line.Split(',');
                    var dataItems = parts.Select(double.Parse).ToArray();
                    yield return Vector.FromArray(dataItems);
                }
            }
        }

        public IEnumerable<string> GetDataSetsNames()
        {
            var files = Directory.GetFiles(this.dataPath.AbsolutePath);
            return files.Select(file => new FileInfo(file).Name);
        }

        private interface ISSVNode<F, X>
        {
            ParentSSVNode<F, X> Parent { get; set; }

            ISSVNode<F, X> Sibling { get; set; }

            decimal SSVW { get; set; }

            X[] Data { get; set; }
        }

        private abstract class BaseSSVNode<F, X> : ISSVNode<F, X>
        {
            public ParentSSVNode<F, X> Parent { get; set; }

            public ISSVNode<F, X> Sibling { get; set; }

            public decimal SSVW { get; set; }

            public X[] Data { get; set; }
        }

        private class ParentSSVNode<F, X> : BaseSSVNode<F, X>
        {
            public ISSVNode<F, X> Left { get; set; }

            public ISSVNode<F, X> Right { get; set; }
        }

        private class LeafSSVNode<F, X> : BaseSSVNode<F, X>
        {
            public F Feature { get; set; }
        }

        private static void CalculateSSVW<F, X>(ISSVNode<F, X> node, Func<IEnumerable<X>, X[][]> Dc)
        {
            if(node.Parent != null && node.Parent.Sibling != null)
            {
                var ps = node.Parent.Sibling;
                var D = node.Data.Union(ps.Data);
                node.SSVW = SSV<X, F>(D, Dc, node.Data, ps.Data);
            }

            var parent = node as ParentSSVNode<F, X>;
            if(parent != null)
            {
                CalculateSSVW<F, X>(parent.Left, Dc);
                CalculateSSVW<F, X>(parent.Right, Dc);
            }
        }

        private static Dictionary<string, decimal> CreateSSVContinuouisificationLookup(string[][] records, int column)
        {
            Func<IEnumerable<string[]>, string[][][]> Dc = xs => xs.Select(x => new[] { x }).ToArray();
            Func<string[], string> f = xs => xs[column];
            var features = records.Select(f).ToArray();
            var tree = GrowTree<string[], string>(records, Dc, f, features);
            ReorderSSVNodes<string, string[]>(tree);
            var leafs = LeafNodes<string, string[]>(tree as ParentSSVNode<string, string[]>).ToArray();
            leafs = NodesWithNeighbouringSSV<string, string[]>(leafs, Dc).ToArray();
            var values = LookupValues<string, string[]>(leafs);
            var result = new Dictionary<string, decimal>();
            for(var i = 0; i < leafs.Length; i++)
            {
                result.Add(leafs[i].Feature, values[i]);
            }

            return result;
        }

        private static decimal[] LookupValues<F, X>(IEnumerable<LeafSSVNode<F, X>> nodes)
        {
            var nodesArray = nodes.ToArray();
            var result = new decimal[nodesArray.Length];
            var denSum = 0M;
            for (var j = 0; j < nodesArray.Length - 1; j++)
            {
                denSum += nodesArray[j].SSVW;
            }

            for (var i = 0; i < nodesArray.Length; i++)
            {
                var numSum = 0M;
                for(var j = 0; j < i - 1; j++)
                {
                    numSum += nodesArray[j].SSVW;
                }

                result[i] = numSum / denSum;
            }

            return result;
        }

        private static IEnumerable<LeafSSVNode<F, X>> NodesWithNeighbouringSSV<F, X>(IEnumerable<LeafSSVNode<F, X>> source, Func<IEnumerable<X>, X[][]> Dc)
        {
            var sourceArray = source.ToArray();
            for (var i = 0; i < sourceArray.Length - 1; i++)
            {
                var left = sourceArray[i];
                var right = sourceArray[i + 1];
                var node = new LeafSSVNode<F, X>
                {
                    Data = left.Data,
                    Feature = left.Feature,
                    SSVW = SSV<X, F>(left.Data.Union(right.Data), Dc, left.Data, right.Data)
                };
                yield return node;
            }

            {
                //TODO What to do with the last entry????
                var left = sourceArray[sourceArray.Length - 2];
                var right = sourceArray[sourceArray.Length - 1];
                var node = new LeafSSVNode<F, X>
                {
                    Data = right.Data,
                    Feature = right.Feature,
                    SSVW = SSV<X, F>(left.Data.Union(right.Data), Dc, left.Data, right.Data)
                };

                yield return node;
            }
        }

        private static IEnumerable<LeafSSVNode<F,X>> LeafNodes<F, X>(ParentSSVNode<F,X> parent)
        {
            var leftLeaf = parent.Left as LeafSSVNode<F,X>;
            if(leftLeaf != null)
            {
                yield return leftLeaf;
            }
            else
            {
                foreach (var leaf in LeafNodes<F, X>(leftLeaf as ParentSSVNode<F, X>))
                    yield return leaf;
            }

            var rightLeaf = parent.Right as LeafSSVNode<F, X>;
            if (rightLeaf != null)
            {
                yield return rightLeaf;
            }
            else
            {
                foreach (var leaf in LeafNodes<F, X>(rightLeaf as ParentSSVNode<F, X>))
                    yield return leaf;
            }
        }

        private static void ReorderSSVNodes<F, X>(ISSVNode<F, X> node)
        {           
            var parent = node as ParentSSVNode<F, X>;
            if (parent != null)
            {
                if (node.Parent != null)
                {
                    if (node == node.Parent.Left)
                    {
                        if(parent.Left.SSVW < parent.Right.SSVW)
                        {
                            SwapSiblings<F, X>(parent);
                        }
                    }

                    if(node == node.Parent.Right)
                    {
                        if(parent.Left.SSVW > parent.Right.SSVW)
                        {
                            SwapSiblings<F, X>(parent);
                        }
                    }
                }

                ReorderSSVNodes<F, X>(parent.Left);
                ReorderSSVNodes<F, X>(parent.Right);
            }
        }

        private static void SwapSiblings<F, X>(ParentSSVNode<F, X> parent)
        {
            var tempNode = parent.Left;
            parent.Left = parent.Right;
            parent.Right = tempNode;
        }

        private static ISSVNode<F, X> GrowTree<X, F>(IEnumerable<X> D, Func<IEnumerable<X>, X[][]> Dc, Func<X, F> f, IEnumerable<F> features)
        {
            var ssvs = features.Select(feature =>
                {
                    var ls = LS(f, feature, D).ToArray();
                    var rs = RS(f, feature, D).ToArray();
                    return new { feature = feature, ssv = SSV<X, F>(D, Dc, ls, rs), ls = ls, rs = rs };
                }
                ).OrderByDescending(t => t.ssv).ToArray();
            var selected = ssvs.First();
            var right = new LeafSSVNode<F, X> 
            {
                Feature = selected.feature,
                Data = selected.rs
            };
            var left = GrowTree<X, F>(selected.ls, Dc, f, features.Except(new[] { selected.feature }).ToArray());
            left.Data = selected.ls;
            left.Sibling = right;
            right.Sibling = left;

            var result = new ParentSSVNode<F, X>
            {
                Right = right,
                Left = left
            };

            left.Parent = result;
            right.Parent = result;

            return result;
        }

        private static decimal SSV<X, F>(IEnumerable<X> D, Func<IEnumerable<X>, X[][]> Dc, IEnumerable<X> ls, IEnumerable<X> rs)
        {
            var nom = Dc(D).Sum(c => ls.Intersect(c).Count() * rs.Except(c).Count());
            var denom = Dc(D).Sum(c => Math.Min(ls.Intersect(c).Count(), rs.Intersect(c).Count()));
            return 2 * nom - denom;
        }

        private static IEnumerable<X> LS<X, F>(Func<X, F> f, F s, IEnumerable<X> data)
        {
            return data.Where(d => !f(d).Equals(s));
        }

        private static IEnumerable<T> RS<T, W>(Func<T, W> f, W s, IEnumerable<T> data)
        {
            return data.Where(d => f(d).Equals(s));
        }
    }
}
