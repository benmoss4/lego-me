import streamlit as st
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy.ndimage import label
import pulp
import math
import boto3

s3 = boto3.client('s3',aws_access_key_id=st.secrets["AWS"]["ACCESS_KEY"],aws_secret_access_key=st.secrets["AWS"]["SECRET_KEY"])

st.title('legome by Ben Moss')

image_data = st.file_uploader('Upload Square Portrait Image',['png','jpg'])

if image_data is not None:
    # To read file as bytes:
    bytes_data = image_data.getvalue()

    nparr = np.fromstring(bytes_data, np.uint8)

    ##image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    ##image = cv2.cvtColor(cv2.imdecode(nparr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    imagedata = pd.DataFrame(cv2.resize(cv2.cvtColor(cv2.imdecode(nparr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (48, 48)).reshape(2304,3)).rename(columns={0: "r", 1:"g", 2:"b"})

    ##image = image.reshape(2304,3)

    imagedata['pixel_id'] = imagedata.index+1
    imagedata['y'] = (imagedata['pixel_id']/48).apply(np.ceil)
    imagedata['x'] = ((imagedata['pixel_id']-1)%48)+1
    imagedata.r = imagedata.r /255
    imagedata.g = imagedata.g /255
    imagedata.b = imagedata.b /255
    
    kmeans = KMeans(n_clusters=5).fit(imagedata[['r','g','b']])
    ##centroids = kmeans.cluster_centers_
    centroidstable = pd.DataFrame(kmeans.cluster_centers_)
    centroidstable.columns = ['cluster_r','cluster_g','cluster_b']
    centroidsarray = kmeans.predict(centroidstable)
    centroidstable['cluster'] = centroidsarray
    grouparray = kmeans.predict(imagedata[['r','g','b']]) 
    imagedata['cluster'] = grouparray
    df_inner = pd.merge(imagedata, centroidstable, on='cluster', how='inner')
    
    labels = {'Group': [], 'y': [], 'x': []}

    labels = pd.DataFrame(data=labels)

    for i in [0,1,2,3,4]:
    
        df_filtery = df_inner['cluster']==i
        df_filtery = df_inner[df_filtery]
    
        df_filtery['cluster_r'] = 1
    
        df_filterx = df_inner['cluster']!=i
        df_filterx = df_inner[df_filterx]
    
        df_filterx['cluster_r'] = 0
    
        frames = [df_filtery, df_filterx]

        result = pd.concat(frames)

        result = result.sort_values(by=['y','x'])

        result = result.drop(columns=['r','g','b','pixel_id','x','y','cluster','cluster_g','cluster_b'])

        almost = result.to_numpy().reshape(48,48)
        array = np.array(almost, dtype=np.uint8)
        
        labeled_array, num_features = label(array)
        all_labels = labeled_array.reshape(2304,1)


        ##blobs = array > 0.7 * array.mean()

        ##all_labels = measure.label(blobs,connectivity=1)

        ##all_labels = all_labels.reshape(2304,1)

        all_labels = pd.DataFrame(all_labels)

        all_labels['RowID'] = (all_labels.index+1)/48
        all_labels['y'] = all_labels['RowID'].apply(np.ceil)
        all_labels['RowID'] = all_labels.index
        all_labels['x'] = (all_labels['RowID']%48)+1

        alllabels_filter = all_labels[0]!=0
        all_labels = all_labels[alllabels_filter]

        all_labels = all_labels.rename(columns={0: "Group"})

        all_labels = all_labels.drop(columns=['RowID'])

        frames = [all_labels, labels]

        labels = pd.concat(frames)
        
    df_inner = pd.merge(df_inner, labels, on=['y','x'], how='inner')
    
    df_inner = df_inner.drop(columns=['r','g','b'])
    
    centroidstable['lightness'] = (centroidstable[['cluster_r','cluster_g','cluster_b']].max(axis=1)-centroidstable[['cluster_r','cluster_g','cluster_b']].min(axis=1))/2
    df_inner = df_inner.sort_values(by=['pixel_id'])
    df_filtery = df_inner['y']==1
    df_filtery = df_inner[df_filtery]
    df_filterx = df_inner['x']==1
    df_filterx = df_inner[df_filterx]
    vertical_stack = pd.concat([df_filterx,df_filtery], axis=0)
    edge_counts = vertical_stack.groupby(['cluster']).size().reset_index(name='counts')
    edge_counts = edge_counts.sort_values(by=['counts'])
    cluster_legocolours = edge_counts.iloc[[-1]]
    cluster_legocolours = pd.DataFrame(cluster_legocolours)

    chosen_colour = st.radio(label = 'Select Background Colour', options = ['Bright Yellow','Bright Blue','Bright Red'])

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    cluster_legocolours['colour'] = chosen_colour
    cluster_legocolours = cluster_legocolours.drop(columns=['counts'])
    other_clusters = pd.merge(cluster_legocolours, centroidstable, on='cluster', how='outer')
    other_clusters = other_clusters[other_clusters['colour'].isnull()]
    other_clusters = other_clusters.drop(columns=['colour'])
    other_clusters = other_clusters.sort_values(by=['lightness'])
    other_clusters = other_clusters.reset_index(drop=True)
    other_clusters['colour_id'] = (other_clusters.index)+1
  
    data = [[1,'Black'], [2, 'Dark Stone Grey'], [3, 'Medium Stone Grey'],[4,'White']]
    lego_colours = pd.DataFrame(data, columns = ['colour_id', 'colour']) 
    other_clusters = pd.merge(other_clusters, lego_colours, on='colour_id', how='outer')
    other_clusters = other_clusters.drop(columns=['lightness','cluster_r','cluster_g','cluster_b','colour_id'])
    cluster_colours = pd.concat([other_clusters,cluster_legocolours], axis=0)
    df_inner = pd.merge(df_inner,cluster_colours, on='cluster', how='inner')
    df_inner = df_inner.drop(columns=['cluster','cluster_r','cluster_g','cluster_b'])

    ##scaffold = pd.read_csv('scaffold.csv')

    obj = s3.get_object(Bucket= 'legome', Key='supportingfiles/scaffold.csv') 
    scaffold = pd.read_csv(obj['Body'])

    lego_colours = pd.DataFrame({'colour': ['Black', 'White', 'Dark Stone Grey','Medium Stone Grey','Bright Yellow','Bright Blue','Bright Red'], 'r': [0,255,91,162,254,0,222], 'g': [0,255,102,170,196,87,0], 'b': [0,255,102,173,0,168,13]})

    withcolours = pd.merge(df_inner, lego_colours, on='colour', how='inner')

    withcolours = pd.merge(withcolours, scaffold, on=['x','y'], how='inner').sort_values(by=['new_y','new_x'])[['r','g','b']]

    withcolours = withcolours.to_numpy().reshape(480,480,3)

    array = np.array(withcolours, dtype=np.uint8)

    image_1 = st.image(image = array)

    option_lego = st.checkbox('Lego Me!')

    if option_lego:

        progress_bar = st.progress(0)

        progress_value = 0

        with st.spinner('Optimizing cost...'):

            ##lego_store = pd.read_csv('lego_store.csv')

            obj = s3.get_object(Bucket= 'legome', Key='supportingfiles/lego_store.csv') 
            lego_store = pd.read_csv(obj['Body'])

            lego_storex = lego_store['region']=='en-gb'
            lego_store = lego_store[lego_storex]
            
            legostorey = lego_store['size']=='BASE PLATE 48X48'
            legostorey = lego_store[legostorey]
            
            region_bricks = lego_store[['brick_id','price','area','colour']]
            # get object and file (key) from bucket

            ##lego_bits = pd.read_csv('lego_bits.csv') # 'Body' is a key word

            obj = s3.get_object(Bucket= 'legome', Key='supportingfiles/lego_bits.csv') 
            lego_bits = pd.read_csv(obj['Body'])

            valid_bits = pd.merge(region_bricks,lego_bits,on = 'brick_id')
            
            ##big_table = pd.merge(valid_bits,df_inner, on=['x','y','colour'], how='right')
            
            big_table = pd.merge(valid_bits,df_inner, on=['x','y','colour'], how='inner')
            
            brick_counts = big_table.groupby(['brick_id','area']).size().reset_index(name='count')
            brick_counts_filter = brick_counts['area']==brick_counts['count']
            brick_counts = brick_counts[brick_counts_filter]
            brick_counts = brick_counts.drop(columns=['area','count'])
            all_possible_bricks = pd.merge(big_table,brick_counts,on=['brick_id'],how='inner')
            
            all_possible_bricks['Group'] = all_possible_bricks['Group'].astype(str)

            all_possible_bricks['colourGroup'] = all_possible_bricks['colour']+"_"+all_possible_bricks['Group']
            
            all_possible_bricks['brick_id'] = 'Brick' + all_possible_bricks['brick_id'].astype(str)
            
            all_possible_bricks['pixel_id'] = 'Pixel' + all_possible_bricks['pixel_id'].astype(str)
            
            list_of_pixels_bricks = pd.DataFrame(all_possible_bricks.groupby(['brick_id','pixel_id']).size().reset_index(name='counts'))

            mylist = all_possible_bricks['colourGroup'].tolist()
            myset = set(mylist)
            mylist = list(myset)
            
            data = []

            incriment = 90/len(mylist)

            for i in mylist:

                progress_bar.progress(math.floor(progress_value))

                progress_value += incriment

                all_possible_bricksx = all_possible_bricks['colourGroup']==i
                these_bricks = all_possible_bricks[all_possible_bricksx]

                these_bricks['flag'] = 1

                newf = these_bricks.pivot(index='brick_id', columns='pixel_id',values='flag').reset_index()

                newf = newf.rename(columns={'pixel_id': 'brick_id'})

                newf = newf.fillna(0)

                prices = lego_store[['brick_id','price']]

                prices['brick_id'] = 'Brick' + prices['brick_id'].astype(str)

                prices = prices.rename(columns={'price': 'Cost'})

                newf = pd.merge(prices,newf,on=['brick_id'],how='inner')

                df = newf

                problem = pulp.LpProblem(name="Calories",sense=pulp.LpMinimize)

                menu_list = df['brick_id'].tolist()

                Cost = df['Cost'].tolist()

                listpixels = df.columns.tolist()

                listpixels.remove('brick_id')
                listpixels.remove('Cost')

                LpVariableList = [pulp.LpVariable('{}'.format(item), cat='Binary') for item in menu_list]

                problem += pulp.lpDot(Cost, LpVariableList)

                for x in listpixels:

                    problem += pulp.lpDot(df[x].tolist(), LpVariableList) == 1

                problem.solve()

                for v in problem.variables():
                    if v.varValue==1: data.append(v.name)
            
            final_bricks = pd.DataFrame(data)

            final_bricks.columns = ['brick_id']
            
            final_bricks_and_pixels = pd.merge(final_bricks, list_of_pixels_bricks, on='brick_id', how='inner')
            final_bricks_and_pixels = final_bricks_and_pixels.drop(columns=['counts'])
            
            df_inner['pixel_id'] = 'Pixel' + df_inner['pixel_id'].astype(str)

            bricks_and_pixel_locations = pd.merge(final_bricks_and_pixels,df_inner,on='pixel_id',how='inner')

            d = {'colour': ['Black', 'White', 'Dark Stone Grey','Medium Stone Grey','Bright Yellow','Bright Blue','Bright Red'], 'r': [0,255,91,162,254,0,222], 'g': [0,255,102,170,196,87,0], 'b': [0,255,102,173,0,168,13]}
            df = pd.DataFrame (d)

            ##Bring the RGB values into the pixel list

            ##withcolours = pd.merge(bricks_and_pixel_locations, df, on='colour', how='inner')

            ##withcoloursnew = pd.merge(withcolours, scaffold, on=['x','y'], how='inner')

            ##Improve quality of image by increasing the number of pixels 10 fold (from 48x48 to 480x480)

            ##withcoloursnew = withcoloursnew.sort_values(by=['new_y','new_x'])

            ##Sort data so that when we generete array pixels are in right position

            ##withcoloursnew = withcoloursnew [['r','g','b']]

            ##Keep only RGB columns

            ##almost = withcoloursnew.to_numpy().reshape(480,480,3)

            ##array = np.array(almost, dtype=np.uint8)

            ##image_1 = st.image(image = cv2.cvtColor(array,cv2.COLOR_RGB2BGR))

            a = bricks_and_pixel_locations.groupby("brick_id").agg({"x":np.min}).reset_index()
            a = a.rename(columns={ a.columns[1]: "Min_x" })
            b = bricks_and_pixel_locations.groupby("brick_id").agg({"x":np.max}).reset_index()
            b = b.rename(columns={ b.columns[1]: "Max_x" })
            c = bricks_and_pixel_locations.groupby("brick_id").agg({"y":np.min}).reset_index()
            c = c.rename(columns={ c.columns[1]: "Min_y" })
            d = bricks_and_pixel_locations.groupby("brick_id").agg({"y":np.max}).reset_index()
            d = d.rename(columns={ d.columns[1]: "Max_y" })
            
            boundaries = pd.merge(a,b, on= "brick_id")
            boundaries = pd.merge(boundaries,c, on= "brick_id")
            boundaries = pd.merge(boundaries,d, on= "brick_id")
            
            boundaries['Min_x'] = (boundaries['Min_x']*10)-10
            boundaries['Max_x'] = (boundaries['Max_x']*10)
            boundaries['Min_y'] = (boundaries['Min_y']*10)-10
            boundaries['Max_y'] = (boundaries['Max_y']*10)
            
            boundaries['Min_x'] = boundaries['Min_x'].astype(int)
            boundaries['Max_x'] = boundaries['Max_x'].astype(int)
            boundaries['Min_y'] = boundaries['Min_y'].astype(int)
            boundaries['Max_y'] = boundaries['Max_y'].astype(int)
            
            image_2 = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
            
            for index, row in boundaries.iterrows():
            ##print(row['Min_x'], row['Max_x'],row['Min_y'], row['Max_y'])
                cv2.rectangle(image_2, (row['Min_x'],row['Min_y']), (row['Max_x'], row['Max_y']), (0,0,0), 1)
            
            x = []
            i = 5
            while i < 480:
                x.append(i)
                i+=10
            
            y = []
            i = 5
            while i < 480:
                y.append(i)
                i+=10
                
            x = pd.DataFrame(data=x)
            y = pd.DataFrame(data=y)
            
            x = x.rename(columns={ x.columns[0]: "x" })
            y = y.rename(columns={ y.columns[0]: "y" })
            
            x['Join'] = "1"
            y['Join'] = "1"
            
            points = pd.merge(x,y, on = "Join")
            
            for index, row in points.iterrows():
            
                cv2.circle(image_2,(row['x'],row['y']), radius=0, color=(0, 0,0), thickness=1)

        progress_bar.progress(100)
   
        image_2 = cv2.cvtColor(image_2,cv2.COLOR_BGR2RGB)
        
        st.image(image = image_2)

        def my_agg(x):
            names = {
                    'quantity': x['brick_id'].nunique(),
                    'price': x['price'].mean(),
                    'total_price':  x['price'].sum()}

            return pd.Series(names, index=['quantity','price','total_price'])
            
        lego_store['brick_id'] = 'Brick' + lego_store['brick_id'].astype(str)

        lego_store = pd.DataFrame(lego_store.groupby(['brick_id','element_id','design_id','size','price','colour','brick_url']).size().reset_index())

        brickgroupby = pd.merge(final_bricks, lego_store, on='brick_id', how='inner')

        brickgroupby = pd.concat([legostorey,brickgroupby], axis=0,sort=False).reset_index()

        brickgroupby['design_id'] = brickgroupby['design_id'].fillna('-')

        costlist = brickgroupby.groupby(['element_id','design_id','colour','size','brick_url']).apply(my_agg).sort_values(by=['colour','size']).reset_index()

        import io
        is_success, buffer = cv2.imencode(".jpg", image_2)
        io_buf = io.BytesIO(buffer)

        st.download_button("Press to Download Cost List",costlist.to_csv().encode('utf-8'),"cost-list.csv","text/csv")

        st.download_button("Press to Download Build Instructions",io_buf,"build-instructions.png","image/png")
