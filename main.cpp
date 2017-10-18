#include <bits/stdc++.h>

using namespace std;
#define ll long long
#define M 1000000007  //1000000009
#define INF 9999999999999 // 9223372036854775807
#define mp(x, y) make_pair(x,y)
#define pb(x) push_back(x)
#define pmp(x, y) pb(mp(x,y))
#define ld double
#define PI 3.14159265358979
#define len(a) (ll)a.size()    //
#define F first
#define S second
#define endl "\n"
#define ALL(x) x.begin() , x.end()
#define B begin()
#define E end()
#define fast() ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0)
#define sz 100010

vector<ll> ans, v;

void add() {
    ans.clear();
    ll carry = 0;
    for (ll i = 5; i >= 0; i--) {
    	
    	ll dig=v[i] + carry + v[i + 6];
        
        ans.push_back(dig%2);
        carry = dig/2;
    
    }

    ans.push_back(carry);
   reverse(ans.begin(), ans.end());

}

int main() {
    srand(time(NULL));

    ofstream data ("data.txt");
    ofstream out ("ans.txt");
    ll n = 3000;
    for (ll i = 0; i < n; i++) {

        v.clear();
        for (ll j = 0; j < 12; j++) {
            v.push_back(rand() % 2);
            data<<v.back()<<" ";
        }
        add();
        data<<endl;
        for(auto &j:ans)
            out<<j<<" ";
        out<<endl;
    }


}
