from flask import Flask, render_template, request
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pandas.plotting import table
app = Flask(__name__)

def analysisofdata(pathoffile, missthreshold=0.3):
    pathoffile = pathoffile.strip('\"')
    try:
        data = pd.read_csv(pathoffile)
    except FileNotFoundError:
        return {"error": f"Error: File '{pathoffile}' not found..."}
    except pd.errors.EmptyDataError:
        return {"error": f"Error: File '{pathoffile}' is empty..."}
    except pd.errors.ParserError:
        return {"error": f"Error: Unable to parse the CSV file '{pathoffile}'. Please check the format..."}

    ############################Downloading the DataFrame##########################################################################
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the figure size as needed

    # Plot the table
    tab = table(ax, data, loc='center', colWidths=[0.2]*len(data.columns))

    # Styling the table
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1.2, 1.2)  # Adjust the scaling as needed

    # Remove axis
    ax.axis('off')

    # Save the image in the static folder
    image_path = os.path.join(os.getcwd(),"static","images",'dataframe_table.png')
    plt.savefig(image_path, bbox_inches='tight')

    ################################################################################################################################
    rows = data.shape[0]
    data = data[data.isnull().sum(axis=1) / data.shape[1] <= missthreshold]
    removed_rows = rows - data.shape[0]

    n_columns = data.select_dtypes(['float64', 'int64']).columns
    for cx in n_columns:
        quar1 = data[cx].quantile(0.25)
        quar3 = data[cx].quantile(0.75)
        IQR = quar3- quar1
        lower_bound = quar1 - 1.5 * IQR
        upper_bound = quar3 + 1.5 * IQR
        
        data[cx] = data[cx].apply(lambda x: x if lower_bound <= x <= upper_bound else None)

    # Call vizofdata with the data
    image_paths = vizofdata(data)

    output_data = {
        "sample_data": data.to_frame().head().to_html(),
        "removed_rows": f"Removed {removed_rows} rows with missing values exceeding {missthreshold * 100}% threshold",
        "overview": data.to_frame().info().to_html(),
        "descriptive_stats": data.to_frame().describe().to_html(),
        "unique_values": data.to_frame().nunique().to_html(),
        "correlation_matrix": data.to_frame().corr().to_html(),
        "image_paths": image_paths
    }
    
    
    return output_data

def vizofdata(data):
    numerical_colms = data.select_dtypes(['float64', 'int64']).columns
    image_paths = []

    for c in numerical_colms:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[c], kde=True)
        plt.title(f"Histogram of {c}")
        plt.xlabel(c)
        plt.ylabel('Freq')

        # Save the image to the static folder
        image_path = f"C:/Users/Vidhut Sharma/Desktop/Flask Folder/EDA_Analysis/static/images/histogram_{c}.png"
        plt.savefig(image_path)
        plt.close()  # Close the figure
        image_paths.append(image_path)

    category_colms = data.select_dtypes(include=['object']).columns
    for f in category_colms:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=data, x=f)
        plt.title(f"Barchart of {f}")
        plt.xlabel(f)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha="right")

        # Save the image to the static/images folder
        image_path = f"C:/Users/Vidhut Sharma/Desktop/Flask Folder/EDA_Analysis/static/images/barchart_{f}.png"
        plt.savefig(image_path) 
        plt.close()  # Close the figure
        image_paths.append(image_path)

    #return image_paths

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    path_of_file = request.form['csvFile']

    # Call your data analysis function
    analysis_result = analysisofdata(path_of_file)
    
    sample_data_html = analysis_result["sample_data"]
    removed_rows = analysis_result["removed_rows"]
    overview_html = analysis_result["overview"]
    descriptive_stats_html = analysis_result["descriptive_stats"]
    unique_values_html = analysis_result["unique_values"]
    correlation_matrix_html = analysis_result["correlation_matrix"]

    # Pass analysis result to the result.html template
    return render_template('result.html', sample_data_html=sample_data_html,
        removed_rows=removed_rows,
        overview_html=overview_html,
        descriptive_stats_html=descriptive_stats_html,
        unique_values_html=unique_values_html,
        correlation_matrix_html=correlation_matrix_html,result=analysis_result)


if __name__ == "__main__":
    # pathoffile = input("Enter the path of your CSV file:- ")
    # print("\n")
    
    # analysisofdata(pathoffile)
    app.run(host = "0.0.0.0",port = 5003)

