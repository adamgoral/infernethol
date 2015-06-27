namespace InferNetHOL.Clustering.Models
{
    public class DatasetColumn
    {
        public DatasetColumn(string name)
        {
            this.Name = name;
        }

        public bool Selected { get; set; }

        public string Name { get; private set; }
    }
}
