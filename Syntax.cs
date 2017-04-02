using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;

namespace MkFn {
    /*
    フィールドの種類
    */
    public enum FieldKind {
        Unknown,
        DomainField,
        CalculatedField,
        ParameterField,
    }

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

        public void AddField(Variable fld) {
            Fields.Add(fld);
            fld.ParentVar = this;
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
            switch (MkFn.OutputLanguage) {
            case Language.CPP:
            case Language.CUDA:
                return Name + "*";

            case Language.CS:
            case Language.MathJax:
                return Name + "[" + new string(',', DimCnt - 1) + "]";
            default:
                throw new Exception();
            }
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

        // フィールドの種類
        public FieldKind Kind = FieldKind.Unknown;

        public bool IsField() {
            return !(this is Function) && ParentVar is Class;
        }

        public Variable(string name, Class type, Term domain, FieldKind kind = FieldKind.Unknown) {
            Name = name;
            TypeVar = type;
            Domain = domain;
            Kind = kind;

            if (Domain != null) {
                Domain.Parent = this;
            }
        }

        /*
            コピーを返します。
        */
        public Variable Clone(Dictionary<Variable, Variable> var_tbl = null) {
            if (this is Function) {
                return (this as Function).Clone(var_tbl);
            }

            if(var_tbl == null) {
                var_tbl = new Dictionary<Variable, Variable>();
            }

            Term domain = (Domain == null ? null : Domain.Clone(var_tbl));
            Variable v1 = new Variable(Name, TypeVar, domain);
            var_tbl.Add(this, v1);

            return v1;
        }

        public virtual string Code() {
            return ToString();
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

        /*
            コピーを返します。
        */
        public new Function Clone(Dictionary<Variable, Variable> var_tbl = null) {
            if (var_tbl == null) {
                var_tbl = new Dictionary<Variable, Variable>();
            }

            Term domain = (Domain == null ? null : Domain.Clone(var_tbl));
            Function fnc = new Function(Name, TypeVar);
            var_tbl.Add(this, fnc);

            fnc.Params = (from x in Params select x.Clone(var_tbl)).ToList();

            fnc.BodyStatement = BodyStatement.Clone(var_tbl);

            return fnc;
        }

        /*
            コンストラクターならtrueを返します。
        */
        public bool IsConstructor() {
            return Name == MkFn.ConstructorName(ParentVar as Class);
        }
    }

    /*
        文
    */
    public class Statement {
        // 親
        public object ParentStmt;

        /*
            コピーを返します。
        */
        public Statement Clone(Dictionary<Variable, Variable> var_tbl = null) {
            if (this is Assignment) {
                // 変数参照の場合

                return (this as Assignment).Clone(var_tbl);
            }
            else if (this is Return) {
                // 数値定数の場合

                return (this as Return).Clone(var_tbl);
            }
            else if (this is ForEach) {
                // 関数適用の場合

                return (this as ForEach).Clone(var_tbl);
            }
            else if (this is BlockStatement) {
                // LINQの場合

                return (this as BlockStatement).Clone(var_tbl);
            }
            else {
                Debug.Assert(false);
                return null;
            }
        }
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

        /*
            コピーを返します。
        */
        public new Assignment Clone(Dictionary<Variable, Variable> var_tbl = null) {
            return new Assignment(Left.Clone(var_tbl), Right.Clone(var_tbl));
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

        /*
            コピーを返します。
        */
        public new Return Clone(Dictionary<Variable, Variable> var_tbl) {
            return new Return(Value == null ? null : Value.Clone(var_tbl));
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

        public void AddStatement(Statement stmt) {
            Statements.Add(stmt);
            stmt.ParentStmt = this;
        }

        /*
            コピーを返します。
        */
        public new BlockStatement Clone(Dictionary<Variable, Variable> var_tbl) {
            return new BlockStatement( (from x in Statements select x.Clone(var_tbl)).ToList() );
        }
    }

    /*
        foreach文
    */
    public class ForEach : BlockStatement {
        // ループ変数
        public List<Variable> LoopVariables = new List<Variable>();

        public ForEach(IEnumerable<Variable> variables, List<Statement> statements) : base(statements) {
            LoopVariables.AddRange(variables);

            foreach(Variable va in variables) {
                va.ParentVar = this;
            }

            foreach (Statement s in Statements) {
                s.ParentStmt = this;
            }
        }

        /*
            コピーを返します。
        */
        public new ForEach Clone(Dictionary<Variable, Variable> var_tbl) {
            if (var_tbl == null) {
                var_tbl = new Dictionary<Variable, Variable>();
            }

            return new ForEach(from va in LoopVariables select va.Clone(var_tbl), (from x in Statements select x.Clone(var_tbl)).ToList());
        }
    }

    /*
        項
    */
    public abstract class Term {
        public static int TermCnt;
        public int TermIdx;

        // 親
        public object Parent;

        // 係数
        public double Value = 1;

        // 項の型
        public Class TypeTerm;

        public Term() {
            TermCnt++;
            TermIdx = TermCnt;
            //if (TermIdx == 1666) {
            //    Debug.Write("");
            //}
        }

        /*
            係数を除く本体が同じならtrueを返します。
        */
        public abstract bool EqBody(Object obj);

        /*
            係数と本体が同じならtrueを返します。
        */
        public virtual bool Eq(Object obj) {
            return obj is Term && Value == (obj as Term).Value && EqBody(obj);
        }

        /*
            コピーを返します。
        */
        public Term Clone(Dictionary<Variable, Variable> var_tbl = null) {
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
            Applyにキャストします。
        */
        public Apply AsApply() {
            return this as Apply;
        }

        /*
            Referenceにキャストします。
        */
        public Reference AsReference() {
            return this as Reference;
        }

        /*
            加算ならtrueを返します。
        */
        public bool IsAdd() {
            return this is Apply && AsApply().Function.VarRef == MkFn.Singleton.AddFnc;
        }

        /*
            乗算ならtrueを返します。
        */
        public bool IsMul() {
            return this is Apply && AsApply().Function.VarRef == MkFn.Singleton.MulFnc;
        }

        public virtual string Code() {
            return ToString();
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
        public virtual int HashCode() {
            return GetHashCode();
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
            コピーを返します。
        */
        public new Number Clone(Dictionary<Variable, Variable> var_tbl) {
            
            return new Number(Value, TypeTerm);
        }

        /*
            係数を除く本体が同じならtrueを返します。
        */
        public override bool EqBody(Object obj) {
            return obj is Number;
        }

        public override string ToStringBody() {
            return Value.ToString();
        }

        public override int HashCode() {
            return GetType().GetHashCode() + (int)Value;
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
            コピーを返します。
        */
        public new Reference Clone(Dictionary<Variable, Variable> var_tbl = null) {
            Variable v1;
            Reference clone_ref;

            if (var_tbl == null || !var_tbl.TryGetValue(VarRef, out v1)) {
                v1 = VarRef;
            }

            if (Indexes == null) {
                clone_ref = new Reference(Name, v1, null, Value);
            }
            else {

                Term[] idx = (from t in Indexes select t.Clone(var_tbl)).ToArray();

                clone_ref = new Reference(Name, v1, idx, Value);
                if (MkFn.Singleton.CloneTable != null && MkFn.Singleton.CloneTable.ContainsKey(this)) {

                    MkFn.Singleton.CloneTable[this] = clone_ref;
                }
            }

            clone_ref.TypeTerm = TypeTerm;
            return clone_ref;
        }

        /*
            係数を除く本体が同じならtrueを返します。
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
            値の定義ならtrueを返します。
        */
        public bool Defined() {
            return Parent is Assignment && (Parent as Assignment).Left == this;
        }

        /*
            値の使用ならtrueを返します。
        */
        public bool Used() {
            return !Defined();
        }

        public override string ToStringBody() {
            if (Indexes == null) {
                return Name;
            }
            else {
                string idx;

                if (MkFn.OutputLanguage == Language.CUDA || MkFn.OutputLanguage == Language.CPP) {

                    idx = MkFn.OffsetFromIndexes(this);


                    string str = Name + "[" + idx + "]";

                    if (MkFn.Singleton.t_var != null && Indexes.Any(x => MkFn.Is_t_plus(x))) {

                        return "(t + 1 < T ? " + str + " : 0)";
                    }
                    else if (MkFn.Singleton.t_var != null && Indexes.Any(x => MkFn.Is_t_minus(x))) {

                        return "(0 <= t - 1 ? " + str + " : 0)";
                    }
                    else {

                        return str;
                    }
                }
                else {

                    idx = string.Join(", ", from x in Indexes select x.ToString());

                    return Name + "[" + idx + "]";
                }

            }
        }

        public override int HashCode() {
            int n = GetType().GetHashCode() + ((int)Value) + Name.GetHashCode();
            
            if(Indexes != null) {
                n += Indexes.Sum(x => x.HashCode());
            }

            return n;
        }
    }

    /*
        関数適用
    */
    public class Apply : Term {
        // 関数
        public Reference Function;

        public Class NewClass;

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

        public Apply(Variable function, Class new_class, params Term[] args) : this(new Reference(function), args) {
            Debug.Assert(new_class != null);
            NewClass = new_class;
        }

        /*
            コピーを返します。
        */
        public new Apply Clone(Dictionary<Variable, Variable> var_tbl = null) {
            Term[] args = (from t in Args select t.Clone(var_tbl)).ToArray();
            Apply app = new Apply(Function.Clone(var_tbl), args);
            app.NewClass = NewClass;
            app.Value = Value;
            app.TypeTerm = TypeTerm;

            return app;
        }

        public override string ToStringBody() {
            if ("+-*/%".Contains(Function.Name[0])) {
                string s;

                Debug.Assert(Args.Length != 1);

                if (IsAdd()) {
                    // 加算の場合

                    //s = string.Join(" ", from x in Args select (x == Args[0] || x.Value < 0 ? "" : "+ ") + (Math.Abs(x.Value) == 1 ? x.ToStringBody() : x.ToString()));
                    s = string.Join(" ", from x in Args select (x == Args[0] || x.Value < 0 ? "" : "+ ") + x.ToString());
                }
                else {
                    // 加算でない場合

                    s = string.Join(" " + Function.Name + " ", from x in Args select x.ToString());
                }

                if (Parent is Apply && (Parent as Apply).Precedence() <= Precedence()) {
                    // 親の演算子の優先順位が高い場合

                    // カッコで囲みます。
                    return "(" + s + ")";
                }
                else {
                    return s;
                }
            }
            else {
                // 演算子でない場合

                if (Function.VarRef == MkFn.Singleton.DiffFnc && Args[0] is Reference && (Args[0] as Reference).VarRef == MkFn.Singleton.EFnc) {

                    return "δ_" + Args[1].ToString();
                }
                else if (Function.VarRef == MkFn.Singleton.MaxPoolPrimeFnc) {
                    Reference x = Args[0] as Reference;
                    Reference y = Args[1] as Reference;

                    string x_idx = MkFn.OffsetFromIndexes(x);
                    string y_idx = MkFn.OffsetFromIndexes(y);

                    return string.Format("({0}[{1}] == {2} ? {3}[{2}] : 0)", MkFn.IndexName(x.VarRef), y_idx, x_idx, x.Name);
                }
                else if (MkFn.IsNew(this)) {

                    return Function.Name + " " + NewClass.Name + "[" + string.Join(", ", from x in Args select x.ToString()) + "]";
                }
                else {

                    return Function.Name + "(" + string.Join(", ", from x in Args select x.ToString()) + ")";
                }
            }
        }

        /*
            係数を除く本体が同じならtrueを返します。
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
            演算子の優先順位を返します。
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

        public override int HashCode() {
            int n = GetType().GetHashCode() + ((int)Value) + Function.HashCode();

            n += (int)Args.Sum(x => (double)x.HashCode());

            return n;
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
            係数を除く本体が同じならtrueを返します。
        */
        public override bool EqBody(Object obj) {
            return this == obj;
        }

        /*
            コピーを返します。
        */
        public new LINQ Clone(Dictionary<Variable, Variable> var_tbl) {
            if (var_tbl == null) {
                var_tbl = new Dictionary<Variable, Variable>();
            }

            Variable[] vars = (from v in Variables select v.Clone(var_tbl)).ToArray();
            LINQ lnq = new LINQ(vars, Select.Clone(var_tbl), (Aggregate == null ? null : Aggregate.Clone(var_tbl)), Value);
            lnq.TypeTerm = TypeTerm;

            return lnq;
        }

        public override string ToStringBody() {
            if (MkFn.LinqValue != null) {

                return MkFn.LinqValue[this];
            }
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
