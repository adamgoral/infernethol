using MicrosoftResearch.Infer.Maths;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InferNetHOL.Clustering.Models
{
    public class InferenceResultItem
    {
        public InferenceResultItem(InferenceResult<Cluster[]> result)
        {
            LogOdds = result.Evidence.LogOdds;
            ClusterCount = result.Result.Length;
            Summary = "Means: " + string.Join(", ", result.Result.Select(c => "[" + ToString(c.Means) + "]"));
            this.Clusters = result.Result;
        }

        private static string ToString(Vector vector)
        {
            return string.Join(",", vector.Select(item => Math.Round(item, 2)));
        }

        public double LogOdds { get; private set; }

        public int ClusterCount { get; private set; }

        public string Summary { get; private set; }

        public Cluster[] Clusters { get; private set; }
    }
}
