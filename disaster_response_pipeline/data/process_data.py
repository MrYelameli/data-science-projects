import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np


def load_data(messages_filepath, categories_filepath):
    '''
    
    load and merge 
    
    Parameters:
    messages_filepath: message (csv file) 
    categories_filepath: categories (csv file) 
    
    Returns:
    df: merged dataframe of messages and dataframe
    
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets on common id and assign to df
    df = messages.merge(categories, how ='outer', on =['id'])
    return df



def clean_data(df):
    '''
    
    Cleans the data
    
    Parameter: 
    df: Dataframe
    
    Return:
    df: Cleaned Dataframe
    
    '''
    categories = df.categories.str.split(pat=';',expand=True)
    firstrow = categories.iloc[0,:]
    category_colnames = firstrow.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
  
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    df.related.replace(2,1,inplace=True)
    df = df.drop_duplicates()
    return df
    

    


def save_data(df, database_filename):
    """Stores df in a SQLite database."""
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('df', engine, index=False)
    pass
    


def main():
    """Loads data, cleans data, saves data to database"""
    print(sys.argv[0])
    if len(sys.argv) == 4:
        

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()