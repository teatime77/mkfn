using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace MkFn {
    public partial class MkFn {

        /*
            添え字から配列の先頭からのオフセットの計算式を得ます。
        */
        public static Term OffsetFromIndexes(Reference rf) {
            if (!IsNew(rf.VarRef.Domain)) {
                // 変数の定義域がない場合

                throw new Exception();
            }

            // 対象の変数参照をコピーします。
            rf = rf.Clone();

            // 変数の定義域をコピーします。
            Apply domain = (rf.VarRef.Domain as Apply).Clone();

            Term t = rf.Indexes[0];
            for (int dim = 1; dim < rf.Indexes.Length; dim++) {
                t = Add(Mul(domain.Args[dim], t), rf.Indexes[dim]);
            }

            return t;
        }


        /*
         
        バッチ
            配列のアクセス

        検証
            CalculatedField
                単一
                複数
            ParameterField
                単一
                複数
            レイヤー
                単一
                複数
         */
    }
}
