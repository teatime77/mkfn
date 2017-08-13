using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;
using System.Reflection;

namespace MkFn {
    public partial class MkFn {
        public static MkFn Singleton;

        string HomeDir = @"C:\usr\prj\mkfn\mkfn";
        public string[] Lines;
        List<Class> SimpleTypes = new List<Class>();
        List<ArrayType> ArrayTypes = new List<ArrayType>();
        List<Class> AppClasses = new List<Class>();

        public int TokenPos;
        public Token CurrentToken;
        Token NextToken;
        
        List<Token> TokenList;
        Token EOTToken = new Token(TokenType.EOT, TokenSubType.Unknown, null, -1, -1);

        Token ReadNextToken() {
            Token current_token = CurrentToken;

            CurrentToken = NextToken;

            // トークンの位置を1つ進めます。
            TokenPos++;

            if (TokenPos + 1 < TokenList.Count) {

                NextToken = TokenList[TokenPos + 1];
            }
            else{

                NextToken = EOTToken;
            }

            return current_token;
        }

        Token GetToken(string text) {
            if(CurrentToken.Text != text) {

                throw new SyntaxException();
            }

            return ReadNextToken();
        }

        Token GetToken(TokenType type) {
            if (type != TokenType.Any && type != CurrentToken.TypeTkn) {

                throw new SyntaxException();
            }

            return ReadNextToken();
        }

        /*
            using文を読みます。
        */
        void ReadUsing() {
            GetToken("using");
            GetToken(TokenType.Identifier);

            while (CurrentToken.Text == ".") {

                GetToken(".");
                GetToken(TokenType.Identifier);
            }

            GetToken(";");

        }

        /*
            修飾子を読みます。
        */
        void ReadModifier() {
            while (true) {
                if(CurrentToken.Text == "public") {
                    GetToken(TokenType.Any);
                }
                else if (CurrentToken.Text == "abstract") {
                    GetToken(TokenType.Any);
                }
                else if (CurrentToken.Text == "static") {
                    GetToken(TokenType.Any);
                }
                else if (CurrentToken.Text == "override") {
                    GetToken(TokenType.Any);
                }
                else if (CurrentToken.Text == "const") {
                    GetToken(TokenType.Any);
                }
                else {
                    break;
                }
            }
        }

        Class GetSimpleType(string name) {

            var v = from c in SimpleTypes where c.Name == name select c;
            if (v.Any()) {

                return v.First();
            }

            Class type = new Class(name);
            SimpleTypes.Add(type);

            return type;
        }

        ArrayType GetArrayType(Class element_type, int dim_cnt) {
            var v = ArrayTypes.Where(c => c.ElementType == element_type && c.DimCnt == dim_cnt);
            if (v.Any()) {

                return v.First();
            }

            ArrayType type = new ArrayType(element_type, dim_cnt);
            ArrayTypes.Add(type);

            return type;
        }

        /*
            型を読みます。
        */
        Class ReadType() {
            Token type_id = GetToken(TokenType.Identifier);

            if (CurrentToken.Text != "[") {
                // 配列でない場合

                return GetSimpleType(type_id.Text);
            }
            else { 
                // 配列の場合

                GetToken("[");

                int dim_cnt = 1;

                while (CurrentToken.Text == ",") {
                    dim_cnt++;
                    GetToken(",");
                }

                GetToken("]");

                Class element_type = GetSimpleType(type_id.Text);

                return GetArrayType(element_type, dim_cnt);
            }
        }

        /*
            変数宣言を読みます。
        */
        Variable ReadVariable() {
            // 型を読みます。
            Class type = ReadType();

            // 変数名を読みます。
            Token id = GetToken(TokenType.Identifier);

            // 変数を返します。
            return new Variable(id.Text, type, null);
        }

        /*
            変数参照を読みます。
        */
        Reference ReadReference() {
            Token id = GetToken(TokenType.Identifier);

            if(CurrentToken.Text != "[") {
                // 配列でない場合

                // 変数参照を返します。
                return new Reference(id.Text, null, null);
            }
            else {
                // 配列の場合

                GetToken("[");

                List<Term> idxes = new List<Term>();
                while (true) {
                    // 配列の添え字の式を読みます。
                    Term idx = Expression();

                    idxes.Add(idx);

                    if (CurrentToken.Text == "]") {
                        break;
                    }
                    GetToken(",");
                }
                GetToken("]");

                // 配列の変数参照を返します。
                return new Reference(id.Text, null, idxes.ToArray());
            }
        }

        /*
            実引数を読みます。
        */
        List<Term> ReadArgs() {
            List<Term> terms = new List<Term>();
            GetToken("(");

            if (CurrentToken.Text != ")") {

                while (true) {
                    // 式を読みます。
                    terms.Add(Expression());

                    if (CurrentToken.Text == ")") {

                        break;
                    }
                    GetToken(",");
                }
            }
            GetToken(")");

            return terms;

        }

        /*
            基本の式を読みます。
        */
        public Term PrimaryExpression() {
            if(CurrentToken.TypeTkn == TokenType.Identifier) {
                // 変数参照を読みます。
                Reference r = ReadReference();

                if(CurrentToken.Text != "(") {

                    // 変数参照を返します。
                    return r;
                }

                // 実引数を読みます。
                List<Term> args = ReadArgs();

                if(r.Name == "Prod") {

                    if(args.Count == 1 && args[0] is LINQ) {
                        // 引数がLINQの場合

                        LINQ lnq = args[0] as LINQ;
                        lnq.Aggregate = new Reference(ProdFnc);
                        lnq.Aggregate.Parent = lnq;

                        // LINQを返します。
                        return lnq;
                    }
                    else {
                        throw new SyntaxException();
                    }
                }

                // 関数適用を返します。
                return new Apply(r, args.ToArray());
            }
            else if (CurrentToken.TypeTkn == TokenType.Number) {
                // 数値の場合

                Token num = GetToken(TokenType.Number);

                // 数値を返します。
                return new Number(num.Text, num.SubType);
            }
            else if (CurrentToken.Text == "(") {

                GetToken("(");
                if (SimpleTypes.Any(x => x.Name == CurrentToken.Text)) {
                    // キャストの場合

                    // 型を読みます。
                    Class type = ReadType();

                    GetToken(")");

                    // 式を読みます。
                    Apply app = Expression() as Apply;

                    // キャストする型をセットします。
                    app.Cast = type;

                    // 関数適用を返します。
                    return app;
                }

                // 式を読みます。
                Term term = Expression();
                GetToken(")");

                if(CurrentToken.Text == ".") {

                    if (!(term is LINQ)) {
                        // LINQでない場合

                        throw new SyntaxException();
                    }

                    LINQ lnq = term as LINQ;

                    GetToken(".");
                    Token id = GetToken(TokenType.Identifier);

                    GetToken("(");
                    GetToken(")");

                    if(id.Text == SumFnc.Name) {

                        // 合計関数の参照を作ります。
                        lnq.Aggregate = new Reference(SumFnc);
                    }
                    else if (id.Text == MaxFnc.Name) {

                        // 最大値関数の参照を作ります。
                        lnq.Aggregate = new Reference(MaxFnc);
                    }
                    else {

                        throw new SyntaxException();
                    }
                    lnq.Aggregate.Parent = lnq;

                    // LINQを返します。
                    return lnq;
                }

                return term;
            }
            else if (CurrentToken.Text == "from") {
                List<Variable> vars = new List<Variable>();

                while (CurrentToken.Text == "from") {

                    GetToken("from");

                    Token id = GetToken(TokenType.Identifier);

                    GetToken("in");

                    // 変数の領域の式を読みます。
                    Term domain = Expression();

                    vars.Add(new Variable(id.Text, null, domain));
                }

                GetToken("select");

                // select式を読みます。
                Term term = Expression();

                // LINQを返します。
                return new LINQ(vars.ToArray(), term, null);
            }
            else if (CurrentToken.Text == "new") {
                List<Term> args = new List<Term>();

                GetToken("new");

                Token type_id = GetToken(TokenType.Identifier);

                Class cls = GetSimpleType(type_id.Text);

                GetToken("[");
                while (true) {

                    // 式を読みます。
                    args.Add( Expression() );
                    if(CurrentToken.Text != ",") {
                        break;
                    }
                    GetToken(",");
                }
                GetToken("]");

                // newの関数適用を返します。
                return new Apply(MkFn.Singleton.NewFnc, cls, args.ToArray());
            }
            else {
                throw new SyntaxException();
            }
        }


        /*
            単項式を読みます。
        */
        public Term UnaryExpression() {
            if (CurrentToken.Text == "-") {
                // 負号の場合

                GetToken("-");

                // 基本の式を読みます。
                Term t1 = PrimaryExpression();

                // 符号を反転します。
                t1.Value *= -1;

                return t1;
            }
            else {

                // 基本の式を読みます。
                return PrimaryExpression();
            }
        }

        /*
            乗算/除算の式を読みます。
        */
        public Term MultiplicativeExpression() {
            // 単項式を読みます。
            Term t1 = UnaryExpression();

            while (CurrentToken.Text == "*" || CurrentToken.Text == "/") {

                // 現在の演算子を保存します。
                string opr = CurrentToken.Text;

                List<Term> args = new List<Term>();
                args.Add(t1);

                while (CurrentToken.Text == opr) {
                    // 現在のトークンが保存した演算子と同じ場合

                    GetToken(opr);

                    // 単項式を読みます。
                    args.Add(UnaryExpression());
                }

                if (opr == "*") {
                    // 乗算の場合

                    t1 = new Apply(new Reference(MulFnc), args.ToArray());
                }
                else {
                    // 除算の場合

                    t1 = new Apply(new Reference(DivFnc), args.ToArray());
                }
            }

            return t1;
        }

        /*
            加算/減算の式を読みます。
        */
        public Term AdditiveExpression() {
            // 乗算/除算の式を読みます。
            Term t1 = MultiplicativeExpression();

            while (CurrentToken.Text == "+" || CurrentToken.Text == "-") {
                // 現在の演算子を保存します。
                string opr = CurrentToken.Text;

                List<Term> args = new List<Term>();
                args.Add(t1);

                while(CurrentToken.Text == opr) {
                    // 現在のトークンが保存した演算子と同じ場合

                    GetToken(opr);

                    // 乗算/除算の式を読みます。
                    args.Add( MultiplicativeExpression() );
                }

                if(opr == "-") {
                    // 減算の場合

                    // 2番目以降の項の符号を反転します。
                    for (int i = 1; i < args.Count; i++) {

                        args[i].Value *= -1;
                    }
                }

                // 加算の関数適用を作ります。
                t1 = new Apply(new Reference(AddFnc), args.ToArray());
            }

            return t1;
        }


        /*
            式を読みます。
        */
        Term Expression() {
            // 加算/減算の式を読みます。
            return AdditiveExpression();
        }

        /*
            ブロック文を読みます。
        */
        List<Statement> ReadBlockStatement() {
            List<Statement> statements = new List<Statement>();
            GetToken("{");

            while(CurrentToken.Text != "}") {
                // 文を読みます。
                statements.Add( ReadStatement() );
            }
            GetToken("}");

            // 文を返します。
            return statements;
        }

        /*
            文を読みます。
        */
        Statement ReadStatement() {
            if(CurrentToken.TypeTkn == TokenType.Identifier) {
                // 変数参照を読みます。
                Reference left = ReadReference();

                GetToken("=");

                // 式を読みます。
                Term right = Expression();

                GetToken(";");

                // 代入文を返します。
                return new Assignment(left, right);
            }
            else if(CurrentToken.Text == "return") {

                GetToken("return");

                // 式を読みます。
                Term value = Expression();

                GetToken(";");

                // return文を返します。
                return new Return(value);
            }
            else if(CurrentToken.Text == "foreach") {

                GetToken("foreach");
                GetToken("(");

                // 変数宣言を読みます。
                Variable v = ReadVariable();

                GetToken("in");

                // 式を読みます。
                v.Domain = Expression();

                v.Domain.Parent = v;
                GetToken(")");

                // ブロック文を読みます。
                List<Statement> stmts = ReadBlockStatement();

                // foreach文を返します。
                return new ForEach(new Variable[] { v }, stmts);
            }
            else {
                throw new SyntaxException();
            }
        }

        /*
            メソッド定義を読みます。
        */
        Function ReadFunction(string name, Class type) {
            Function fnc = new Function(name, type);

            GetToken("(");

            if (CurrentToken.Text != ")") {

                while (true) {
                    // 変数宣言を読みます。
                    Variable arg = ReadVariable();

                    fnc.Params.Add(arg);

                    if (CurrentToken.Text == ")") {

                        break;
                    }
                    GetToken(",");
                }
            }
            GetToken(")");

            if (CurrentToken.Text == ";") {

                GetToken(";");
            }
            else {

                // ブロック文を読みます。
                fnc.BodyStatement = new BlockStatement(ReadBlockStatement());

                fnc.BodyStatement.ParentStmt = fnc;
            }

            return fnc;
        }

        /*
            クラス定義を読みます。
        */
        Class ReadClass() {
            // 修飾子を読みます。
            ReadModifier();

            GetToken("class");
            Token class_id = GetToken(TokenType.Identifier);
            Class cls = GetSimpleType(class_id.Text);
            
            if (CurrentToken.Text == ":") {
                // スーパークラスがある場合

                GetToken(":");
                Token super_class = GetToken(TokenType.Identifier);

                cls.Parent = GetSimpleType(super_class.Text);
            }
            GetToken("{");

            while (CurrentToken.Text != "}") {

                // 修飾子を読みます。
                ReadModifier();

                // 型を読みます。
                Class type = ReadType();

                if (CurrentToken.Text == "(") {
                    // コンストラクターの場合

                    Function fnc = ReadFunction(MkFn.ConstructorName(cls), type);
                    cls.Functions.Add(fnc);
                    fnc.ParentVar = cls;
                }
                else {
                    // コンストラクターでない場合

                    Token id = GetToken(TokenType.Identifier);

                    if (CurrentToken.Text == ";" || CurrentToken.Text == "=") {
                        // フィールドの場合

                        Term init = null;

                        if (CurrentToken.Text == "=") {
                            // フィールドの初期値がある場合

                            GetToken("=");

                            // 初期値の式を読みます。
                            init = Expression();
                        }

                        GetToken(";");

                        // フィールドを作ります。
                        Variable field = new Variable(id.Text, type, init);

                        // クラスのフィールドのリストに追加します。
                        cls.Fields.Add(field);
                        field.ParentVar = cls;

                        //Debug.WriteLine("field {0}", field.ToString(), "");
                    }
                    else {
                        // メソッドの場合

                        // メソッドを作ります。
                        Function fnc = ReadFunction(id.Text, type);

                        // クラスのメソッドのリストに追加します。
                        cls.Functions.Add(fnc);
                        fnc.ParentVar = cls;
                    }
                }
            }

            GetToken("}");

            return cls;
        }

        /*
            ソースファイルの構文解析をします。
        */
        void ParseSource() {
            TokenPos = 0;
            CurrentToken = TokenList[0];
            NextToken = TokenList[1];

            while (CurrentToken.Text == "using") {
                // 現在のトークンがusingの場合

                // using文を読みます。
                ReadUsing();
            }

            while (CurrentToken != EOTToken) {

                // クラス定義を読みます。
                Class cls = ReadClass();
                AppClasses.Add(cls);
            }
        }

        /*
            ソースファイルを読み、字句解析と構文解析をします。
        */
        public void ReadLexParseSourceFile(string path) {
            // ソースファイルを読みます。、字句解析と構文解析をします。
            Lines = File.ReadAllLines(path, Encoding.UTF8);

            // すべての行の字句解析をします。
            TokenList = new List<Token>();
            for (int i = 0; i < Lines.Length; i++) {
                TokenList.AddRange(RemoveComment( LexicalAnalysis(Lines[i], i) ));
            }

            // ソースファイルの構文解析をします。
            ParseSource();
        }

        /*
            メイン処理
        */
        public void Main() {
            HomeDir = Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName( Assembly.GetExecutingAssembly().Location ) ) );

            // ソースファイルを読み、字句解析と構文解析をします。
            ReadLexParseSourceFile(HomeDir + @"\sample\NeuralNetwork.cs");

            // Layerクラス
            Class layer = (from cls in AppClasses where cls.Name == "Layer" select cls).First();

            // 損失関数
            EFnc = (from fld in layer.Fields where fld.Name == "E" select fld).First();

            // Range関数
            RangeFnc = (from f in layer.Functions where f.Name == "Range" select f).First();

            // シグモイド関数の微分
            σ_prime = (from fnc in layer.Functions where fnc.Name == "σ_prime" select fnc).First();

            // tanhの微分
            tanh_prime = (from fnc in layer.Functions where fnc.Name == "tanh_prime" select fnc).First();

            // レイヤークラスのリスト
            Layers = (from cls in AppClasses where cls.IsSubClass(layer) select cls).ToArray();

            // 名前解決
            ResolveName();

            // 型推論
            foreach (Class cls in Layers) {

                TypeInference(cls);
            }

            DeepLearning();
        }
    }
}
