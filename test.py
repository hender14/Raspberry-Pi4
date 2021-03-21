from socket import *

##################
# 受信側プログラム#
##################

# 受信側アドレスの設定
# 受信側IP
#SrcIP = "54.248.64.253"
SrcIP = "0.0.0.0"
# 受信側ポート番号                           
SrcPort = 8888
# 受信側アドレスをtupleに格納
SrcAddr = (SrcIP, SrcPort)
# バッファサイズ指定
BUFSIZE = 1024

# ソケット作成
udpServSock = socket(AF_INET, SOCK_DGRAM)
# 受信側アドレスでソケットを設定
udpServSock.bind(SrcAddr)

# While文を使用して常に受信待ちのループを実行
while True:                         
     # ソケットにデータを受信した場合の処理
    # 受信データを変数に設定
    data, addr = udpServSock.recvfrom(BUFSIZE) 
    data2, addr = udpServSock.recvfrom(BUFSIZE) 
              # 受信データと送信アドレスを出力
    print(data.decode(), addr)
    print(data2.decode(), addr)

    direction = 'straight'
    direction = direction.encode('utf-8')

    udpServSock.sendto(direction, addr)

    data3, addr = udpServSock.recvfrom(BUFSIZE) 

    print(data3.decode(), addr)
