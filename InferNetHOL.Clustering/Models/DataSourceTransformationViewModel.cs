using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InferNetHOL.Clustering.Models
{
    public abstract class BindableModelBase : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;

        protected void OnPropertyChanged(string propertyName)
        {
            var handler = this.PropertyChanged;
            if (handler != null)
            {
                handler(this, new PropertyChangedEventArgs(propertyName));
            }
        }
    }

    public class DataSourceTransformationViewModel : BindableModelBase
    {
        private DataTable dataSource;

        private SourceColumn[] sourceColumns;

        public DataTable DataSource
        {
            get { return this.dataSource; }
            set
            {
                this.dataSource = value;
                this.OnPropertyChanged("DataSource");
                this.UpdateSourceColumnsList();
            }
        }

        public SourceColumn[] SourceColumns
        {
            get { return this.sourceColumns; }
            private set 
            {
                this.UnbindSourceColumnHandlers(this.sourceColumns);
                this.sourceColumns = value;
                this.BindSourceColumnHandlers(this.sourceColumns);
                this.OnPropertyChanged("SourceColums");
                this.UpdateTransformedColumns();
            }
        }

        private void BindSourceColumnHandlers(IEnumerable<SourceColumn> columns)
        {
            if (columns == null) return;
            foreach (var column in columns)
            {
                column.PropertyChanged += SourceColumn_PropertyChanged;
            }
        }

        private void SourceColumn_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            this.UpdateTransformedColumns();
        }

        private void UpdateTransformedColumns()
        {
            throw new NotImplementedException();
        }

        private void UnbindSourceColumnHandlers(IEnumerable<SourceColumn> columns)
        {
            if (columns == null) return;
            foreach(var column in columns)
            {
                column.PropertyChanged -= SourceColumn_PropertyChanged;
            }
        }

        private void UpdateSourceColumnsList()
        {
            var columns = new List<SourceColumn>();
            foreach (var column in this.dataSource.Columns.Cast<DataColumn>())
            {
                if(IsNumeric(column.DataType))
                {
                    columns.Add(new NumberSourceColumn(column.ColumnName));
                }
                else if(IsTextual(column.DataType))
                {
                    columns.Add(new TextSourceColumn(column.ColumnName));
                }
                else
                {
                    columns.Add(new UnsupportedSourceColumn(column.ColumnName, column.DataType));
                }
            }

            this.SourceColumns = columns.ToArray();
        }

        private static bool IsTextual(Type type)
        {
            return type == typeof(string);
        }

        private static bool IsNumeric(Type type)
        {
            return type == typeof(int) ||
                   type == typeof(double) ||
                   type == typeof(decimal) ||
                   type == typeof(float) ||
                   type == typeof(long);
        }
    }

    public enum TextSourceColumnTransformationTypes
    {
        None,
        Ordinal,
        IdentityVector,
        BagOfWords
    }

    public abstract class SourceColumn : BindableModelBase
    {
        public SourceColumn(string name)
        {
            this.Name = name;
        }

        public string Name { get; private set; }
    }

    public class UnsupportedSourceColumn : SourceColumn
    {
        public UnsupportedSourceColumn(string name, Type dataType)
            : base(name)
        {
            this.DataType = dataType;
        }

        public Type DataType { get; private set; }
    }

    public abstract class SupportedSourceColumn : SourceColumn
    {
        private bool selected;

        public SupportedSourceColumn(string name)
            : base(name)
        {
        }

        public bool Selected
        {
            get { return this.selected; }
            set
            {
                if (this.selected != value)
                {
                    this.selected = value;
                    this.OnPropertyChanged("Selected");
                }
            }
        }
    }

    public class NumberSourceColumn : SupportedSourceColumn
    {
        public NumberSourceColumn(string name)
            : base(name)
        {
        }
    }

    public class TextSourceColumn : SupportedSourceColumn
    {
        private TextSourceColumnTransformationTypes transformation;

        public TextSourceColumn(string name)
            : base(name)
        {
            this.transformation = TextSourceColumnTransformationTypes.Ordinal;
        }

        public TextSourceColumnTransformationTypes Transformation
        {
            get { return this.transformation; }
            set
            {
                if (this.transformation != value)
                {
                    this.transformation = value;
                    this.OnPropertyChanged("Transformation");
                }
            }
        }

        public IEnumerable<TextSourceColumnTransformationTypes> AvailableTransformationTypes
        {
            get
            {
                return Enum.GetValues(typeof(TextSourceColumnTransformationTypes)).Cast<TextSourceColumnTransformationTypes>();
            }
        }
    }
}
