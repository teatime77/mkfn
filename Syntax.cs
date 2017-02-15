using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;

namespace MkFn {
    /*
        クラス
    */
    public class Class {
        // 親
        public Class Parent;

        // クラス名
        public string Name;

        // フィールドのリスト
        public List<Variable> Fields = new List<Variable>();

        // 関数のリスト
        public List<Function> Functions = new List<Function>();

        // 次元 (スカラーは0,  1次元配列は1,  2次元配列は2, ... )
        public int DimCnt;

        public Class(string name) {
            Name = name;
        }

        public virtual Class ValueType() {
            return this;
        }

        public override string ToString() {
            return Name;
        }

        public bool IsSubClass(Class cls) {
            return Parent == cls;
        }
    }

    /*
        配列の型
    */
    public class ArrayType : Class {
        // 要素の型
        public Class ElementType;

        public ArrayType(Class element_type, int dim_cnt) : base(element_type.Name) {
            ElementType = element_type;
            DimCnt = dim_cnt;
        }

        public override Class ValueType() {
            return ElementType;
        }

        public override string ToString() {
            return Name + "[" + new string(',', DimCnt - 1) + "]";
        }
    }

    /*
        変数
    */
    public class Variable {
        // 親
        public object ParentVar;

        // 変数名
        public string Name;

        // 変数の型
        public Class TypeVar;

        // 定義域
        public Term Domain;

        public Variable(string name, Class type, Term domain) {
            Name = name;
            TypeVar = type;
            Domain = domain;

            if (Domain != null) {
                Domain.Parent = this;
            }
        }

        /*
            コピーを返す。
        */
        public Variable Clone(Dictionary<Variable, Variable> var_tbl) {
            Term domain = (Domain == null ? null : Domain.Clone(var_tbl));
            Variable v1 = new Variable(Name, TypeVar, domain);
            var_tbl.Add(this, v1);

            return v1;
        }

        public override string ToString() {
            if (TypeVar == null) {
                return Name;
            }
            return TypeVar.ToString() + " " + Name;
        }
    }

    /*
        関数
    */
    public class Function : Variable {
        // 仮引数
        public List<Variable> Params = new List<Variable>();

        // 関数の本体
        public BlockStatement BodyStatement;

        public Function(string name, Class type) : base(name, type, null) {
        }
    }

    /*
        文
    */
    public class Statement {
        // 親
        public object ParentStmt;
    }

    /*
        代入文
    */
    public class Assignment : Statement {
        // 左辺
        public Reference Left;

        // 右辺
        public Term Right;

        public Assignment(Reference left, Term right) {
            Left = left;
            Right = right;

            Left.Parent = this;
            Right.Parent = this;
        }

        public override string ToString() {
            return Left.ToString() + " = " + Right.ToString();
        }
    }

    /*
        return文
    */
    public class Return : Statement {
        // 戻り値
        public Term Value;

        public Return(Term value) {
            Value = value;
        }
    }

    /*
        ブロック文
    */
    public class BlockStatement : Statement {
        // 文のリスト
        public List<Statement> Statements;

        public BlockStatement(List<Statement> statements) {
            Statements = statements;

            foreach (Statement s in Statements) {
                s.ParentStmt = this;
            }
        }
    }

    /*
        foreach文
    */
    public class ForEach : BlockStatement {
        // ループ変数
        public Variable LoopVariable;

        public ForEach(Variable variable, List<Statement> statements) : base(statements) {
            LoopVariable = variable;

            LoopVariable.ParentVar = this;
            foreach (Statement s in Statements) {
                s.ParentStmt = this;
            }
        }
    }

    /*
        項
    */
    public abstract class Term {
        // 親
        public object Parent;

        // 係数
        public double Value = 1;

        // 項の型
        public Class TypeTerm;

        public Term() {
        }

        /*
            係数を除く本体が同じならtrueを返す。
        */
        public abstract bool EqBody(Object obj);

        /*
            係数と本体が同じならtrueを返す。
        */
        public virtual bool Eq(Object obj) {
            return obj is Term && Value == (obj as Term).Value && EqBody(obj);
        }

        /*
            コピーを返す。
        */
        public Term Clone(Dictionary<Variable, Variable> var_tbl = null) {
            if (var_tbl == null) {
                var_tbl = new Dictionary<Variable, Variable>();
            }

            if (this is Reference) {
                // 変数参照の場合

                return (this as Reference).Clone(var_tbl);
            }
            else if (this is Number) {
                // 数値定数の場合

                return (this as Number).Clone(var_tbl);
            }
            else if (this is Apply) {
                // 関数適用の場合

                return AsApply().Clone(var_tbl);
            }
            else if (this is LINQ) {
                // LINQの場合

                return (this as LINQ).Clone(var_tbl);
            }
            else {
                Debug.Assert(false);
                return null;
            }
        }

        public Term Minus() {
            Value *= -1;

            return this;
        }

        /*
            Applyにキャストする。
        */
        public Apply AsApply() {
            return this as Apply;
        }

        /*
            Referenceにキャストする。
        */
        public Reference AsReference() {
            return this as Reference;
        }

        /*
            加算ならtrueを返す。
        */
        public bool IsAdd() {
            return this is Apply && AsApply().Function.VarRef == MkFn.Singleton.AddFnc;
        }

        /*
            乗算ならtrueを返す。
        */
        public bool IsMul() {
            return this is Apply && AsApply().Function.VarRef == MkFn.Singleton.MulFnc;
        }

        /*
            newならtrueを返す。
        */
        public bool IsNew() {
            return this is Apply && AsApply().Function.Name == "new";
        }

        /*
            Rangeならtrueを返す。
        */
        public static bool IsRange(Term t) {
            return t is Apply && t.AsApply().Function.VarRef == MkFn.Singleton.RangeFnc;
        }

        public override string ToString() {
            if (!(this is Number)) {
                // 数値定数でない場合

                if (Value == 1) {

                    return ToStringBody();
                }
                else if (Value == -1) {

                    return "- " + ToStringBody();
                }
                else {

                    return Value.ToString() + " * " + ToStringBody();
                }
            }
            else {

                return ToStringBody();
            }
        }

        public abstract string ToStringBody();

        public List<Reference> AllRefs() {
            List<Reference> all_refs = new List<Reference>();

            MkFn.Traverse(this,
                delegate (object obj) {
                    if (obj is Reference) {
                        // 変数参照の場合

                        all_refs.Add(obj as Reference);
                    }
                });

            return all_refs;
        }
    }

    /*
        数値定数
    */
    public class Number : Term {
        public Number(string text, TokenSubType sub_type) {
            switch (sub_type) {
            case TokenSubType.Integer:
                int n;
                if (!int.TryParse(text, out n)) {
                    throw new SyntaxException();
                }
                Value = n;
                TypeTerm = MkFn.Singleton.IntClass;
                break;

            case TokenSubType.Float:
                float f;
                if (!float.TryParse(text, out f)) {
                    throw new SyntaxException();
                }
                Value = f;
                TypeTerm = MkFn.Singleton.FloatClass;
                break;

            case TokenSubType.Double:
                if (!double.TryParse(text, out Value)) {
                    throw new SyntaxException();
                }
                TypeTerm = MkFn.Singleton.DoubleClass;
                break;
            }
        }

        public Number(double d, Class type = null) {
            Value = d;

            if (type == null) {
                Debug.Assert(Value == Math.Floor(Value));

                TypeTerm = MkFn.Singleton.IntClass;
            }
            else {

                TypeTerm = type;
            }
        }

        /*
            コピーを返す。
        */
        public new Number Clone(Dictionary<Variable, Variable> var_tbl) {
            
            return new Number(Value, TypeTerm);
        }

        /*
            係数を除く本体が同じならtrueを返す。
        */
        public override bool EqBody(Object obj) {
            return obj is Number;
        }

        public override string ToStringBody() {
            return Value.ToString();
        }
    }

    /*
        変数参照
    */
    public class Reference : Term {
        // 変数名
        public string Name;

        // 参照している変数
        public Variable VarRef;

        // 配列の添え字
        public Term[] Indexes;

        public Reference(string name, Variable ref_var, Term[] idx, double val = 1) {
            Name = name;
            VarRef = ref_var;
            Indexes = idx;
            Value = val;

            if (Indexes != null) {
                foreach (Term t in Indexes) {
                    t.Parent = this;
                }
            }
        }

        public Reference(Variable v, double val = 1) {
            Name = v.Name;
            VarRef = v;
            Indexes = null;
        }

        /*
            コピーを返す。
        */
        public new Reference Clone(Dictionary<Variable, Variable> var_tbl = null) {
            if (var_tbl == null) {
                var_tbl = new Dictionary<Variable, Variable>();
            }
            Variable v1;
            if (!var_tbl.TryGetValue(VarRef, out v1)) {
                v1 = VarRef;
            }

            if (Indexes == null) {
                return new Reference(Name, v1, null, Value);
            }

            Term[] idx = (from t in Indexes select t.Clone(var_tbl)).ToArray();

            return new Reference(Name, v1, idx, Value);
        }

        /*
            係数を除く本体が同じならtrueを返す。
        */
        public override bool EqBody(Object obj) {
            if (!(obj is Reference)) {
                // 変数参照でない場合

                return false;
            }
            Reference r = obj as Reference;
            if (r.VarRef != VarRef) {
                return false;
            }
            if ((Indexes == null) != (r.Indexes == null)) {
                return false;
            }
            if (Indexes == null) {
                return true;
            }
            else {
                Debug.Assert(Indexes.Length == r.Indexes.Length);
                for (int i = 0; i < Indexes.Length; i++) {
                    if (!Indexes[i].Eq(r.Indexes[i])) {
                        return false;
                    }
                }
                return true;
            }
        }

        /*
            値の定義ならtrueを返す。
        */
        public bool Defined() {
            return Parent is Assignment && (Parent as Assignment).Left == this;
        }

        /*
            値の使用ならtrueを返す。
        */
        public bool Used() {
            return !Defined();
        }

        public override string ToStringBody() {
            if (Indexes == null) {
                return Name;
            }
            else {
                return Name + "[" + string.Join(", ", from x in Indexes select x.ToString()) + "]";
            }
        }
    }

    /*
        関数適用
    */
    public class Apply : Term {
        // 関数
        public Reference Function;

        // 引数
        public Term[] Args;

        public Apply(Reference function, Term[] args) {
            Function = function;
            Args = args;

            Function.Parent = this;
            foreach (Term t in Args) {
                t.Parent = this;
            }
        }

        public Apply(Variable function, params Term[] args) : this(new Reference(function), args) {
        }

        /*
            コピーを返す。
        */
        public new Apply Clone(Dictionary<Variable, Variable> var_tbl = null) {
            Term[] args = (from t in Args select t.Clone(var_tbl)).ToArray();
            Apply app = new Apply(Function.Clone(var_tbl), args);
            app.Value = Value;

            return app;
        }

        public override string ToStringBody() {
            if ("+-*/%".Contains(Function.Name[0])) {
                string s;

                Debug.Assert(Args.Length != 1);

                if (IsAdd()) {

                    //s = string.Join(" ", from x in Args select (x == Args[0] || x.Value < 0 ? "" : "+ ") + (Math.Abs(x.Value) == 1 ? x.ToStringBody() : x.ToString()));
                    s = string.Join(" ", from x in Args select (x == Args[0] || x.Value < 0 ? "" : "+ ") + x.ToString());
                }
                else {

                    s = string.Join(" " + Function.Name + " ", from x in Args select x.ToString());
                }

                if (Parent is Apply && (Parent as Apply).Precedence() <= Precedence()) {
                    return "(" + s + ")";
                }
                else {
                    return s;
                }
            }
            else {

                if (Function.VarRef == MkFn.Singleton.DiffFnc && Args[0] is Reference && (Args[0] as Reference).VarRef == MkFn.Singleton.EFnc) {

                    return "δ_" + Args[1].ToString();
                }
                else {

                    return Function.Name + "(" + string.Join(", ", from x in Args select x.ToString()) + ")";
                }
            }
        }

        /*
            係数を除く本体が同じならtrueを返す。
        */
        public override bool EqBody(Object obj) {
            if (!(obj is Apply)) {
                // 関数適用でない場合

                return false;
            }

            Apply app = obj as Apply;

            if ( ! Function.Eq(app.Function) ) {
                // 関数が違う場合

                return false;
            }

            if (Args.Length != app.Args.Length) {
                // 引数の数が違う場合

                return false;
            }

            for (int i = 0; i < Args.Length; i++) {
                if (!Args[i].Eq(app.Args[i])) {
                    // 引数が違う場合

                    return false;
                }
            }

            return true;
        }

        /*
            演算子の優先順位を返す。
        */
        public int Precedence() {
            if (Char.IsLetter(Function.Name[0])) {
                return 20;
            }

            if ("*/%".Contains(Function.Name[0])) {
                return 1;
            }

            if (Args.Length == 1 && "+-!".Contains(Function.Name[0])) {
                return 2;
            }

            return 10;
        }
    }

    /*
        LINQ
    */
    public class LINQ : Term {
        // ループ変数
        public Variable[] Variables;

        // select句
        public Term Select;

        // 集計関数
        public Reference Aggregate;

        public LINQ(Variable[] variables, Term select, Reference aggregate, double val = 1) {
            Variables = variables;
            Select = select;
            Aggregate = aggregate;
            Value = val;

            foreach (Variable v in Variables) {
                v.ParentVar = this;
            }
            Select.Parent = this;
            if (Aggregate != null) {

                Aggregate.Parent = this;
            }
        }

        /*
            係数を除く本体が同じならtrueを返す。
        */
        public override bool EqBody(Object obj) {
            return this == obj;
        }

        /*
            コピーを返す。
        */
        public new LINQ Clone(Dictionary<Variable, Variable> var_tbl) {
            Variable[] vars = (from v in Variables select v.Clone(var_tbl)).ToArray();
            return new LINQ(vars, Select.Clone(var_tbl), (Aggregate == null ? null : Aggregate.Clone(var_tbl)), Value);
        }

        public override string ToStringBody() {
            string list = string.Join(" ", from x in Variables select "from " + x.Name + " in " + x.Domain.ToString()) + " select " + Select.ToString();
            if (Aggregate == null) {

                return list;
            }
            return "(" + list + ")." + Aggregate.Name + "()";
        }
    }

    /*
        構文エラー
    */
    public class SyntaxException : Exception {

        public SyntaxException() {
            Token tkn = MkFn.Singleton.CurrentToken;
            string line = MkFn.Singleton.Lines[tkn.LineIndex];
            Debug.WriteLine("Err {0}行: {1} ^ {2}", tkn.LineIndex + 1, line.Substring(0, tkn.CharPos), line.Substring(tkn.CharPos));
            Debug.Write("");
        }
    }
}
