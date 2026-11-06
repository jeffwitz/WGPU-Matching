struct Input { values: array<f32>, }
struct Output { values: array<f32>, }
@group(0) @binding(0) var<storage,read> packed: Input;
@group(0) @binding(1) var<storage,read_write> delta: Output;

@compute @workgroup_size(1)
fn solve() {
    var augmented:array<array<f32,9>,8>;
    var k=0u;
    for (var row=0u;row<8u;row=row+1u) {
        for (var col=row;col<8u;col=col+1u) {
            let value=packed.values[k]; k=k+1u;
            augmented[row][col]=value; augmented[col][row]=value;
        }
    }
    for (var row=0u;row<8u;row=row+1u) {
        let diagonal=augmented[row][row];
        augmented[row][row]=diagonal+max(abs(diagonal),1.0)*1e-7;
        augmented[row][8]=packed.values[36u+row];
    }
    for (var col=0u;col<8u;col=col+1u) {
        var pivot=col; var magnitude=abs(augmented[col][col]);
        for (var row=col+1u;row<8u;row=row+1u) {
            let candidate=abs(augmented[row][col]);
            if (candidate>magnitude) { magnitude=candidate; pivot=row; }
        }
        if (magnitude<1e-20) { for (var i=0u;i<8u;i=i+1u) { delta.values[i]=0.0; } return; }
        if (pivot!=col) {
            for (var j=col;j<9u;j=j+1u) {
                let tmp=augmented[col][j]; augmented[col][j]=augmented[pivot][j]; augmented[pivot][j]=tmp;
            }
        }
        let divisor=augmented[col][col];
        for (var j=col;j<9u;j=j+1u) { augmented[col][j]=augmented[col][j]/divisor; }
        for (var row=0u;row<8u;row=row+1u) {
            if (row!=col) {
                let factor=augmented[row][col];
                for (var j=col;j<9u;j=j+1u) { augmented[row][j]=augmented[row][j]-factor*augmented[col][j]; }
            }
        }
    }
    for (var i=0u;i<8u;i=i+1u) { delta.values[i]=augmented[i][8]; }
}
