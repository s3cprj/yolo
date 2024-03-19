# yolo
実行ファイル化する前のpythonコード  
要件：yolov5の実行環境  
<https://github.com/ultralytics/yolov5>  
  
・実行方法   
`python detected.py`  
実行結果は、mydata/output.csvに出力される。  
※detected.pyのLOGGER.info()部分のコメントアウトを外してコマンドラインへの出力も可。  
requiments.txtは元々pythonの要件定義が記述してあったが、exe化すると要件を満たしていない判定となりAutoUpdateでパッケージのインストールが実行されてしまう。
それではまずいので、一旦requiments.txtを空白にして要件をスルーした。  
pythonのファイル内でチェックを行っている関数を消そうとしたが、パッケージの中で呼び出されてる部分とかまでカバーする時間が無いので一旦これで着地。
