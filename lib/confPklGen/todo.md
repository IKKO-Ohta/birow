### Configuration pickle Generate libs

## pickleの対応表
oracle2vecが要求するpklの一覧

 - tag_map.pkl
  - 単語と品詞の対応付け
- word2id.pkl
    - 単語のフラッグid
- act_map.pkl
 - parserアクションのidづけ
- tag2id.pkl
 - POS tagのid
 - word2idに依存する

## 今後のtodo
 応急処置として、word2id.pklにword2id["NONE"] = -1 を追加する
 空のときに-1を宛てる作業をbuffer側でしかしてないので、stack側でもするようにする
 欠損しているword2idGen.pyを補う。以前は対話モードから直接生成したのだと思われる
 pickleをoracle形式のみから抽出できるようにする（互換のため）
