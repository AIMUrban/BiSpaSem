the format of raw data for HD and LX

```
adname,cityname,location,name,type
海淀区,北京市,116.359019#40.00583,中国农业大学东校区篮球场,体育休闲服务;运动场馆;篮球场馆
```

the format of raw data for BK(NY) and TKY(TYO)

```
name,latitude,longitude,category_name
Boudier Grocery,40.680321594647964,-73.86917931909201,Grocery Store
```



For type, we choose "篮球场馆" as the category of the POI. That is to say, the last element in the sequence separated by ";" in the "type" is taken as the actual category of a POI.

Note that if precise GPS coordinates are required, please convert the coordinates (GCJ-02) in the raw data of HD and LX.